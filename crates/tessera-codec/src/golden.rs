// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The cases behind `spec/vectors/codec/`, which `cargo xtask gen-vectors` writes and every TCE
//! implementation checks:
//! - golden encodings, each with its value in a language-neutral form, so that an implementation
//!   that only decodes can compare what it read;
//! - inputs each decoder must reject with a given code;
//! - inputs exactly at the limits, which it must accept;
//! - envelopes, each with the payload a reader must return or the code and offset it must
//!   report, pinning the order of RFC-0003 'Envelope'.
//!
//! Field types add their cases in `tessera-field`.

use crate::envelope::{self, DIGEST_LEN, HEADER_LEN, MAGIC};
use crate::error::{ErrorCode, SchemaId};
use crate::schema::{self, Class, Schema};
use crate::{Ascii, Decode, DecodeError, DecodeLimits, Encode, Map, Rational, decode, encode};

/// A value as the vectors write it, as JSON:
/// - integers are decimal strings, so 64-bit values survive JSON;
/// - `Option` is `null` or `{"some": value}`;
/// - an enum value is `{"Variant": fields}`;
/// - a map is its entries in canonical order, as `[key, value]` pairs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Shown {
    /// An integer, in decimal.
    Int(String),
    /// A boolean.
    Bool(bool),
    /// An absent value or a variant without fields.
    Null,
    /// Text.
    Text(String),
    /// A sequence: arrays, vectors, tuples, map entries, tuple-variant fields.
    List(Vec<Shown>),
    /// Named members: struct fields, an enum variant, a fraction's parts.
    Fields(Vec<(&'static str, Shown)>),
}

impl Shown {
    /// The JSON text.
    pub fn json(&self) -> String {
        match self {
            Self::Int(n) => format!("\"{n}\""),
            Self::Bool(b) => b.to_string(),
            Self::Null => "null".to_owned(),
            Self::Text(t) => {
                let escaped: String = t
                    .chars()
                    .flat_map(|c| match c {
                        '"' | '\\' => vec!['\\', c],
                        c => vec![c],
                    })
                    .collect();
                format!("\"{escaped}\"")
            }
            Self::List(items) => {
                let items: Vec<String> = items.iter().map(Self::json).collect();
                format!("[{}]", items.join(","))
            }
            Self::Fields(fields) => {
                let fields: Vec<String> = fields
                    .iter()
                    .map(|(name, value)| format!("\"{name}\":{}", value.json()))
                    .collect();
                format!("{{{}}}", fields.join(","))
            }
        }
    }
}

/// A value the vectors can show.
pub trait Show {
    /// The value, as the vectors write it.
    fn show(&self) -> Shown;
}

macro_rules! show_integers {
    ($($t:ty),*) => {$(
        impl Show for $t {
            fn show(&self) -> Shown {
                Shown::Int(self.to_string())
            }
        }
    )*};
}

show_integers!(u8, u16, u32, u64, i8, i16, i32, i64);

impl Show for bool {
    fn show(&self) -> Shown {
        Shown::Bool(*self)
    }
}

impl<T: Show, const N: usize> Show for [T; N] {
    fn show(&self) -> Shown {
        Shown::List(self.iter().map(Show::show).collect())
    }
}

impl<T: Show> Show for Vec<T> {
    fn show(&self) -> Shown {
        Shown::List(self.iter().map(Show::show).collect())
    }
}

impl<T: Show> Show for Option<T> {
    fn show(&self) -> Shown {
        match self {
            None => Shown::Null,
            Some(value) => Shown::Fields(vec![("some", value.show())]),
        }
    }
}

impl<A: Show, B: Show> Show for (A, B) {
    fn show(&self) -> Shown {
        Shown::List(vec![self.0.show(), self.1.show()])
    }
}

impl Show for Ascii {
    fn show(&self) -> Shown {
        Shown::Text(self.as_str().to_owned())
    }
}

impl Show for Rational {
    fn show(&self) -> Shown {
        Shown::Fields(vec![("num", self.num().show()), ("den", self.den().show())])
    }
}

impl<K: Show, V: Show> Show for Map<K, V> {
    fn show(&self) -> Shown {
        Shown::List(
            self.entries()
                .iter()
                .map(|(k, v)| Shown::List(vec![k.show(), v.show()]))
                .collect(),
        )
    }
}

/// The sample struct of the vectors: one field of each kind.
#[derive(Clone, Debug, PartialEq, Eq, Encode, Decode)]
pub struct Sample {
    /// A `bool`.
    pub flag: bool,
    /// A byte.
    pub small: u8,
    /// A sequence.
    pub list: Vec<u16>,
    /// An optional signed integer.
    pub maybe: Option<i32>,
    /// Text.
    pub name: Ascii,
    /// A fixed-length array.
    pub digest: [u8; 32],
}

impl Show for Sample {
    fn show(&self) -> Shown {
        Shown::Fields(vec![
            ("flag", self.flag.show()),
            ("small", self.small.show()),
            ("list", self.list.show()),
            ("maybe", self.maybe.show()),
            ("name", self.name.show()),
            ("digest", self.digest.show()),
        ])
    }
}

/// The sample enum: unit, tuple, struct and recursive variants.
#[derive(Clone, Debug, PartialEq, Eq, Encode, Decode)]
pub enum Shape {
    /// No fields: tag 0.
    Empty,
    /// One unnamed field: tag 1.
    Point(u32),
    /// Named fields: tag 2.
    Box {
        /// The width.
        width: u16,
        /// The height.
        height: u16,
    },
    /// Recursion, which nesting limits bound: tag 3.
    Nested(Vec<Shape>),
}

impl Shape {
    /// `levels` shapes, each nesting the next, around an empty one.
    pub fn nest(levels: usize) -> Self {
        (0..levels).fold(Self::Empty, |inner, _| Self::Nested(vec![inner]))
    }
}

impl Show for Shape {
    fn show(&self) -> Shown {
        let (variant, fields) = match self {
            Self::Empty => ("Empty", Shown::Null),
            Self::Point(x) => ("Point", Shown::List(vec![x.show()])),
            Self::Box { width, height } => (
                "Box",
                Shown::Fields(vec![("width", width.show()), ("height", height.show())]),
            ),
            Self::Nested(inner) => ("Nested", Shown::List(vec![inner.show()])),
        };
        Shown::Fields(vec![(variant, fields)])
    }
}

// Depth probes: each composite kind nested to a known depth, so that the limit pins how much
// each kind contributes (RFC-0003 'Limits').
type Opt2<T> = Option<Option<T>>;
type Opt4<T> = Opt2<Opt2<T>>;
type Opt8<T> = Opt4<Opt4<T>>;
type Opt32<T> = Opt8<Opt8<Opt8<Opt8<T>>>>;
type Arr2<T> = [[T; 1]; 1];
type Arr4<T> = Arr2<Arr2<T>>;
type Arr8<T> = Arr4<Arr4<T>>;
type Pair2<T> = ((T, u8), u8);
type Pair4<T> = Pair2<Pair2<T>>;
type Pair8<T> = Pair4<Pair4<T>>;

/// `levels` `Some` tags, then `inner`.
fn somes(levels: usize, inner: &[u8]) -> Vec<u8> {
    [[1u8, 0].repeat(levels), inner.to_vec()].concat()
}

/// The schemas of the sample types, as the vectors describe them.
pub const SCHEMAS: [(&str, &str); 2] = [
    (
        "Sample",
        "struct { flag: bool, small: u8, list: Vec<u16>, maybe: Option<i32>, name: Ascii, digest: [u8; 32] }",
    ),
    (
        "Shape",
        "enum { Empty, Point(u32), Box { width: u16, height: u16 }, Nested(Vec<Shape>) }",
    ),
];

/// The limits golden vectors decode under: wide enough for every case.
pub const GOLDEN_LIMITS: DecodeLimits = DecodeLimits {
    max_total_bytes: 1 << 20,
    max_vec_len: 1 << 16,
    max_depth: 32,
};

/// The limits the corpus decodes under: tight enough that limit cases stay small.
pub const CORPUS_LIMITS: DecodeLimits = DecodeLimits {
    max_total_bytes: 64,
    max_vec_len: 16,
    max_depth: 8,
};

/// A value and its encoding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Golden {
    /// The type, as the vectors name it.
    pub ty: &'static str,
    /// The value.
    pub value: Shown,
    /// The encoding.
    pub bytes: Vec<u8>,
}

impl Golden {
    /// The case for `value` of type `ty`.
    pub fn of<T: Encode + Show>(ty: &'static str, value: T) -> Self {
        Self {
            ty,
            value: value.show(),
            bytes: encode(&value),
        }
    }
}

/// An input that decoding as `ty` must reject with `code`, for exactly one reason.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rejected {
    /// The type, as the vectors name it.
    pub ty: &'static str,
    /// The input.
    pub bytes: Vec<u8>,
    /// The code the decoder must return.
    pub code: ErrorCode,
    /// What is wrong with it.
    pub why: &'static str,
}

/// An input exactly at a limit, which decoding as `ty` must accept.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Boundary {
    /// The type, as the vectors name it.
    pub ty: &'static str,
    /// The input.
    pub bytes: Vec<u8>,
    /// Which limit it meets.
    pub why: &'static str,
}

fn le(values: &[u64], width: usize) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| v.to_le_bytes()[..width].to_vec())
        .collect()
}

fn map<K: Encode, V>(entries: Vec<(K, V)>) -> Map<K, V> {
    Map::from_entries(entries).expect("distinct keys")
}

fn ascii(text: &str) -> Ascii {
    Ascii::new(text).expect("printable ASCII")
}

fn rational(num: i64, den: u64) -> Rational {
    Rational::new(num, den).expect("a non-zero denominator")
}

/// A sample whose encoding is `52 + 2·list.len() + name.len()` bytes.
fn sample(list: Vec<u16>, name: &str) -> Sample {
    Sample {
        flag: true,
        small: 1,
        list,
        maybe: None,
        name: ascii(name),
        digest: [0; 32],
    }
}

/// `levels` shapes, each nesting the next, around an empty list: depth `2·levels`.
fn nest_open(levels: usize) -> Shape {
    (1..levels).fold(Shape::Nested(vec![]), |inner, _| Shape::Nested(vec![inner]))
}

/// The golden cases of the codec's own types.
pub fn cases() -> Vec<Golden> {
    let digest: [u8; 32] = core::array::from_fn(|i| i as u8);
    vec![
        Golden::of("u8", 0u8),
        Golden::of("u8", 0xFFu8),
        Golden::of("u16", 0u16),
        Golden::of("u16", 0x1234u16),
        Golden::of("u16", u16::MAX),
        Golden::of("u32", 0xDEAD_BEEFu32),
        Golden::of("u64", 0x0102_0304_0506_0708u64),
        Golden::of("u64", u64::MAX),
        Golden::of("i8", -1i8),
        Golden::of("i16", i16::MIN),
        Golden::of("i32", i32::MIN),
        Golden::of("i32", -2i32),
        Golden::of("i32", i32::MAX),
        Golden::of("i64", i64::MIN),
        Golden::of("i64", -0x0102_0304_0506_0708i64),
        Golden::of("i64", i64::MAX),
        Golden::of("bool", false),
        Golden::of("bool", true),
        Golden::of("Digest", digest),
        Golden::of("[u16; 3]", [1u16, 0x0100, 0xFFFF]),
        Golden::of("Vec<u32>", Vec::<u32>::new()),
        Golden::of("Vec<u32>", vec![7u32, 0x0102_0304, u32::MAX]),
        Golden::of("Bytes", b"\x00\x01\xFE\xFF".to_vec()),
        Golden::of("Vec<i8>", vec![i8::MIN, -1, 0, 1, i8::MAX]),
        Golden::of(
            "Vec<Vec<i16>>",
            vec![vec![], vec![-1i16, 2], vec![i16::MAX]],
        ),
        Golden::of("Ascii", ascii("")),
        Golden::of("Ascii", ascii("tessera/v1 ~!\"\\")),
        Golden::of("Rational", rational(0, 1)),
        Golden::of("Rational", rational(-3, 7)),
        Golden::of("Rational", rational(i64::MIN, 1)),
        Golden::of("Rational", rational(1, u64::MAX)),
        Golden::of("Option<u64>", None::<u64>),
        Golden::of("Option<u64>", Some(5u64)),
        Golden::of("(u8, Bytes)", (9u8, vec![1u8, 2])),
        // Keys sort by encoding: 256 (00 01) comes before 1 (01 00).
        Golden::of("Map<u16, u8>", map(vec![(1u16, 10u8), (256, 20), (2, 30)])),
        Golden::of("Map<u16, u8>", Map::<u16, u8>::default()),
        // A key's length prefix sorts first, little-endian: a 256-byte key comes before a
        // one-byte key, and "b" before "aa".
        Golden::of(
            "Map<Bytes, u8>",
            map(vec![
                (b"aa".to_vec(), 1u8),
                (b"b".to_vec(), 2),
                (vec![0x41; 256], 3),
            ]),
        ),
        Golden::of(
            "Sample",
            Sample {
                flag: true,
                small: 200,
                list: vec![3, 1, 4],
                maybe: Some(-5),
                name: ascii("sample"),
                digest,
            },
        ),
        Golden::of("Shape", Shape::Empty),
        Golden::of("Shape", Shape::Point(42)),
        Golden::of(
            "Shape",
            Shape::Box {
                width: 3,
                height: 4,
            },
        ),
        Golden::of("Shape", Shape::nest(3)),
    ]
}

/// The non-canonical corpus of the codec's own types, under [`CORPUS_LIMITS`]. Every entry
/// fails for one reason only, so its code does not depend on the order of checks, except where
/// RFC-0003 'Limits' fixes that order.
pub fn corpus() -> Vec<Rejected> {
    use ErrorCode::{
        CodecLimit as Limit, CodecNoncanonical as Noncanonical, CodecTruncated as Truncated,
    };
    let bad = |ty, bytes: Vec<u8>, code, why| Rejected {
        ty,
        bytes,
        code,
        why,
    };
    let rational = |num: i64, den: u64| [num.to_le_bytes(), den.to_le_bytes()].concat();
    let ascii = |bytes: &[u8]| [&(bytes.len() as u64).to_le_bytes()[..], bytes].concat();
    let too_long = encode(&sample(vec![7; 5], "abc"));
    assert_eq!(too_long.len(), 65);
    vec![
        bad(
            "bool",
            vec![2],
            Noncanonical,
            "a bool byte other than 0 or 1",
        ),
        bad(
            "bool",
            vec![0xFF],
            Noncanonical,
            "a bool byte other than 0 or 1",
        ),
        bad("u16", vec![1], Truncated, "one byte of two"),
        bad("u64", vec![0; 7], Truncated, "seven bytes of eight"),
        bad(
            "u8",
            vec![1, 2],
            Noncanonical,
            "a byte left over after the value",
        ),
        bad("Ascii", ascii(&[0x1F]), Noncanonical, "a control character"),
        bad("Ascii", ascii(&[0x7F]), Noncanonical, "DEL"),
        bad("Ascii", ascii(&[0x80]), Noncanonical, "a byte above 0x7E"),
        bad("Ascii", ascii(&[b'o', b'k', 0]), Noncanonical, "a NUL byte"),
        bad(
            "Ascii",
            [le(&[3], 8), b"ab".to_vec()].concat(),
            Truncated,
            "two bytes of a declared three",
        ),
        bad(
            "Rational",
            rational(1, 0),
            Noncanonical,
            "a zero denominator",
        ),
        bad(
            "Rational",
            rational(2, 4),
            Noncanonical,
            "a fraction not in lowest terms",
        ),
        bad(
            "Rational",
            rational(0, 2),
            Noncanonical,
            "zero not written 0/1",
        ),
        bad(
            "Rational",
            rational(-3, 9),
            Noncanonical,
            "a negative fraction not in lowest terms",
        ),
        bad(
            "Rational",
            rational(i64::MIN, 2),
            Noncanonical,
            "i64::MIN over an even denominator",
        ),
        bad(
            "Option<u64>",
            le(&[2], 2),
            Noncanonical,
            "an Option tag above 1",
        ),
        bad(
            "Option<u64>",
            le(&[0xFFFF], 2),
            Noncanonical,
            "an Option tag of 0xFFFF",
        ),
        bad(
            "Option<u64>",
            le(&[1], 2),
            Truncated,
            "Some without its payload",
        ),
        bad(
            "Shape",
            le(&[4], 2),
            Noncanonical,
            "an enum tag past the last variant",
        ),
        bad(
            "Shape",
            le(&[2, 3], 2),
            Truncated,
            "a struct variant missing a field",
        ),
        bad(
            "Map<u16, u8>",
            [le(&[2], 8), le(&[1], 2), vec![10], le(&[256], 2), vec![20]].concat(),
            Noncanonical,
            "keys in numeric order, which is not encoding order",
        ),
        bad(
            "Map<u16, u8>",
            [le(&[2], 8), le(&[7], 2), vec![1], le(&[7], 2), vec![2]].concat(),
            Noncanonical,
            "a duplicate key",
        ),
        bad(
            "Map<Bytes, u8>",
            [
                le(&[2], 8),
                le(&[2], 8),
                b"aa".to_vec(),
                vec![1],
                le(&[1], 8),
                b"b".to_vec(),
                vec![2],
            ]
            .concat(),
            Noncanonical,
            "keys in content order, which is not encoding order: the length prefix sorts first",
        ),
        bad(
            "Vec<u8>",
            [le(&[17], 8), vec![0; 17]].concat(),
            Limit,
            "a length above max_vec_len, with every byte present",
        ),
        bad(
            "Vec<u64>",
            [le(&[10], 8), vec![0; 16]].concat(),
            Truncated,
            "a length whose elements cannot fit in the bytes left",
        ),
        bad(
            "Vec<u64>",
            le(&[1 << 63], 8),
            Limit,
            "a length of 2^63: max_vec_len is checked before the bytes left",
        ),
        bad(
            "Sample",
            too_long,
            Limit,
            "a valid encoding one byte above max_total_bytes",
        ),
        bad(
            "Shape",
            encode(&Shape::nest(4)),
            Limit,
            "nesting at depth 9: each Shape and each Vec is one level",
        ),
        bad(
            "Option^9<u8>",
            somes(9, &[7]),
            Limit,
            "nine Options: each is one level",
        ),
        bad(
            "[u8; 1]^9",
            vec![7],
            Limit,
            "nine nested arrays: each is one level",
        ),
        bad(
            "pair^9",
            vec![7; 10],
            Limit,
            "nine nested pairs: each is one level",
        ),
        bad(
            "Option^8<Ascii>",
            somes(8, &[le(&[1], 8), b"a".to_vec()].concat()),
            Limit,
            "eight Options around Ascii, which is one level",
        ),
        bad(
            "Option^8<Rational>",
            somes(8, &rational(1, 2)),
            Limit,
            "eight Options around a Rational, which is one level",
        ),
        bad(
            "Option^7<Map<u8, u8>>",
            somes(7, &[le(&[1], 8), vec![1, 2]].concat()),
            Limit,
            "seven Options around a map entry: the map and the entry are one level each",
        ),
    ]
}

/// Inputs exactly at the limits of [`CORPUS_LIMITS`], which every decoder must accept.
pub fn boundaries() -> Vec<Boundary> {
    let at = |ty, bytes, why| Boundary { ty, bytes, why };
    let exact = encode(&sample(vec![7; 5], "ab"));
    assert_eq!(exact.len(), 64);
    vec![
        at(
            "Vec<u8>",
            [le(&[16], 8), vec![0; 16]].concat(),
            "a length equal to max_vec_len",
        ),
        at("Sample", exact, "an input of exactly max_total_bytes"),
        at(
            "Shape",
            encode(&nest_open(4)),
            "nesting at exactly max_depth: four shapes around an empty list",
        ),
        at("Option^8<u8>", somes(8, &[7]), "eight Options: depth 8"),
        at("[u8; 1]^8", vec![7], "eight nested arrays: depth 8"),
        at("pair^8", vec![7; 9], "eight nested pairs: depth 8"),
        at(
            "Option^7<Ascii>",
            somes(7, &[le(&[1], 8), b"a".to_vec()].concat()),
            "seven Options and an Ascii: depth 8",
        ),
        at(
            "Option^7<Rational>",
            somes(7, &[1i64.to_le_bytes(), 2u64.to_le_bytes()].concat()),
            "seven Options and a Rational: depth 8",
        ),
        at(
            "Option^6<Map<u8, u8>>",
            somes(6, &[le(&[1], 8), vec![1, 2]].concat()),
            "six Options, a map and one entry: depth 8",
        ),
        at(
            "Vec<Option<u64>>",
            [le(&[16], 8), le(&[0; 16], 2)].concat(),
            "sixteen sibling Options at depth 2: depth is released between siblings",
        ),
    ]
}

/// What decoding `bytes` as the type named `ty` gives: the value shown and encoded again, or
/// the error. `None` for a type this crate does not define.
pub type Outcome = Option<Result<(Shown, Vec<u8>), DecodeError>>;

/// Decodes `bytes` as the type named `ty`.
pub fn decode_as(ty: &str, bytes: &[u8], limits: DecodeLimits) -> Outcome {
    fn read<T: Decode + Encode + Show>(
        bytes: &[u8],
        limits: DecodeLimits,
    ) -> Result<(Shown, Vec<u8>), DecodeError> {
        decode::<T>(bytes, limits).map(|value| (value.show(), encode(&value)))
    }
    Some(match ty {
        "u8" => read::<u8>(bytes, limits),
        "u16" => read::<u16>(bytes, limits),
        "u32" => read::<u32>(bytes, limits),
        "u64" => read::<u64>(bytes, limits),
        "i8" => read::<i8>(bytes, limits),
        "i16" => read::<i16>(bytes, limits),
        "i32" => read::<i32>(bytes, limits),
        "i64" => read::<i64>(bytes, limits),
        "bool" => read::<bool>(bytes, limits),
        "Digest" => read::<[u8; 32]>(bytes, limits),
        "[u16; 3]" => read::<[u16; 3]>(bytes, limits),
        "Vec<u8>" | "Bytes" => read::<Vec<u8>>(bytes, limits),
        "Vec<i8>" => read::<Vec<i8>>(bytes, limits),
        "Vec<u32>" => read::<Vec<u32>>(bytes, limits),
        "Vec<u64>" => read::<Vec<u64>>(bytes, limits),
        "Vec<Vec<i16>>" => read::<Vec<Vec<i16>>>(bytes, limits),
        "Ascii" => read::<Ascii>(bytes, limits),
        "Rational" => read::<Rational>(bytes, limits),
        "Option<u64>" => read::<Option<u64>>(bytes, limits),
        "(u8, Bytes)" => read::<(u8, Vec<u8>)>(bytes, limits),
        "Map<u16, u8>" => read::<Map<u16, u8>>(bytes, limits),
        "Map<Bytes, u8>" => read::<Map<Vec<u8>, u8>>(bytes, limits),
        "Sample" => read::<Sample>(bytes, limits),
        "Shape" => read::<Shape>(bytes, limits),
        "Option^8<u8>" => read::<Opt8<u8>>(bytes, limits),
        "Option^9<u8>" => read::<Option<Opt8<u8>>>(bytes, limits),
        "[u8; 1]^8" => read::<Arr8<u8>>(bytes, limits),
        "[u8; 1]^9" => read::<[Arr8<u8>; 1]>(bytes, limits),
        "pair^8" => read::<Pair8<u8>>(bytes, limits),
        "pair^9" => read::<(Pair8<u8>, u8)>(bytes, limits),
        "Option^7<Ascii>" => read::<Opt4<Opt2<Option<Ascii>>>>(bytes, limits),
        "Option^8<Ascii>" => read::<Opt8<Ascii>>(bytes, limits),
        "Option^7<Rational>" => read::<Opt4<Opt2<Option<Rational>>>>(bytes, limits),
        "Option^8<Rational>" => read::<Opt8<Rational>>(bytes, limits),
        "Option^6<Map<u8, u8>>" => read::<Opt4<Opt2<Map<u8, u8>>>>(bytes, limits),
        "Option^7<Map<u8, u8>>" => read::<Opt4<Opt2<Option<Map<u8, u8>>>>>(bytes, limits),
        "Vec<Option<u64>>" => read::<Vec<Option<u64>>>(bytes, limits),
        "Option^32<u8>" => read::<Opt32<u8>>(bytes, limits),
        "Option^33<u8>" => read::<Option<Opt32<u8>>>(bytes, limits),
        _ => return None,
    })
}

// ----- Envelopes ------------------------------------------------------------------------------

/// The schema the envelope vectors expect: outside the catalog, at version 3, so that its readers
/// accept versions 3 and 2 and the vectors can show both.
pub const PROBE_SCHEMA: Schema = Schema {
    name: "EnvelopeProbe",
    id: SchemaId(0xFFF0),
    version: 3,
    extension: None,
    class: Class::Public,
};

/// The limits most envelope vectors are read under.
pub const ENVELOPE_LIMITS: DecodeLimits = DecodeLimits {
    max_total_bytes: 256,
    max_vec_len: 256,
    max_depth: 8,
};

/// The limits an envelope case is read under: the vectors' own, or one of RFC-0003's defaults,
/// by name, so that every implementation checks its own constants against the vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaseLimits {
    /// [`ENVELOPE_LIMITS`].
    Vectors,
    /// The verifier-message defaults: 8 GiB, 2^34 elements, 32 levels.
    Message,
    /// The `IntegerModel` defaults: 1 TiB, otherwise as for messages.
    IntegerModel,
}

impl CaseLimits {
    /// The name the vectors give these limits, `None` for the vectors' own.
    pub fn name(self) -> Option<&'static str> {
        match self {
            Self::Vectors => None,
            Self::Message => Some("message"),
            Self::IntegerModel => Some("integer-model"),
        }
    }
}

/// RFC-0003 'Limits', as the RFC writes them: what [`CaseLimits::Message`] and
/// [`CaseLimits::IntegerModel`] must be.
pub const RFC_MESSAGE_TOTAL: u64 = 8 << 30;
/// See [`RFC_MESSAGE_TOTAL`].
pub const RFC_MESSAGE_VEC_LEN: u64 = 1 << 34;
/// See [`RFC_MESSAGE_TOTAL`].
pub const RFC_MESSAGE_DEPTH: u16 = 32;
/// See [`RFC_MESSAGE_TOTAL`].
pub const RFC_INTEGER_MODEL_TOTAL: u64 = 1 << 40;

/// An envelope that a reader expecting `expect` checks under `limits`, then decodes as `ty`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnvelopeCase {
    /// The schema the reader expects, at the version its writers emit.
    pub expect: Schema,
    /// The limits it reads under.
    pub limits: CaseLimits,
    /// The payload's type, as the vectors name it.
    pub ty: &'static str,
    /// The input.
    pub bytes: Vec<u8>,
    /// The payload the reader returns, or the code and offset it reports.
    pub outcome: Result<Vec<u8>, (ErrorCode, u64)>,
    /// What the case shows.
    pub why: &'static str,
}

/// An envelope's fields, each free to be wrong, with a digest that matches whatever they hold.
#[derive(Clone)]
struct Fields {
    magic: [u8; 4],
    envelope_version: u16,
    schema: u16,
    version: u16,
    reserved: u16,
    payload_len: Option<u64>,
    payload: Vec<u8>,
}

impl Fields {
    fn of(payload: Vec<u8>) -> Self {
        Self {
            magic: MAGIC,
            envelope_version: envelope::ENVELOPE_VERSION,
            schema: PROBE_SCHEMA.id.0,
            version: PROBE_SCHEMA.version,
            reserved: 0,
            payload_len: None,
            payload,
        }
    }

    fn header(&self) -> Vec<u8> {
        let payload_len = self.payload_len.unwrap_or(self.payload.len() as u64);
        [
            &self.magic[..],
            &self.envelope_version.to_le_bytes(),
            &self.schema.to_le_bytes(),
            &self.version.to_le_bytes(),
            &self.reserved.to_le_bytes(),
            &payload_len.to_le_bytes(),
        ]
        .concat()
    }

    fn bytes(&self) -> Vec<u8> {
        let mut bytes = [self.header(), self.payload.clone()].concat();
        let digest = blake3::Hasher::new_derive_key(envelope::DIGEST_CONTEXT)
            .update(&bytes)
            .finalize();
        bytes.extend_from_slice(digest.as_bytes());
        bytes
    }
}

/// `bytes` with bit 0 of the byte at `at` flipped.
fn flipped(mut bytes: Vec<u8>, at: usize) -> Vec<u8> {
    bytes[at] ^= 1;
    bytes
}

/// The envelope cases, in groups:
/// - accepted envelopes;
/// - one rejection per check of RFC-0003 'Envelope', each with exactly one fault, header fields
///   cut short at every offset among them;
/// - pairs of faults, each pinning which of two checks comes first;
/// - RFC-0003's default limits, exactly at and one past each bound.
pub fn envelopes() -> Vec<EnvelopeCase> {
    use ErrorCode::{
        CodecDigest as Digest, CodecLimit as Limit, CodecMagic as Magic,
        CodecNoncanonical as Noncanonical, CodecTruncated as Truncated, CodecVersion as Version,
    };
    let case = |limits, bytes, ty, outcome, why| EnvelopeCase {
        expect: PROBE_SCHEMA,
        limits,
        ty,
        bytes,
        outcome,
        why,
    };
    let ok = |fields: &Fields, ty, why| {
        case(
            CaseLimits::Vectors,
            fields.bytes(),
            ty,
            Ok(fields.payload.clone()),
            why,
        )
    };
    let bad = |bytes, code, offset, why| {
        case(
            CaseLimits::Vectors,
            bytes,
            "Vec<u64>",
            Err((code, offset)),
            why,
        )
    };
    let base = Fields::of(encode(&vec![1u64, 2]));
    let whole = base.bytes();
    let digest_at = (HEADER_LEN + base.payload.len()) as u64;
    let end = whole.len() as u64;
    let with = |edit: &dyn Fn(&mut Fields)| {
        let mut fields = base.clone();
        edit(&mut fields);
        fields.bytes()
    };
    // The largest `Bytes` payload that keeps the envelope within the limits: its length prefix,
    // then the bytes.
    let room = ENVELOPE_LIMITS.max_total_bytes as usize - HEADER_LEN - DIGEST_LEN - 8;
    let fits = Fields::of(encode(&vec![0x5Au8; room]));
    let over = Fields::of(encode(&vec![0x5Au8; room + 1]));
    let previous = Fields {
        version: PROBE_SCHEMA.version - 1,
        ..base.clone()
    };
    let undecodable = Fields::of([encode(&vec![7u64]), vec![0]].concat());
    let mut cases = vec![
        ok(&base, "Vec<u64>", "the current version"),
        ok(
            &previous,
            "Vec<u64>",
            "the previous version: readers accept v and v - 1",
        ),
        ok(
            &Fields::of(encode(&Vec::<u64>::new())),
            "Vec<u64>",
            "an empty sequence",
        ),
        ok(&fits, "Bytes", "exactly max_total_bytes in all"),
        // The header, field by field: each field cut short, then each field wrong.
        bad(Vec::new(), Truncated, 0, "nothing at all"),
        bad(b"TSR".to_vec(), Truncated, 0, "three bytes of the magic"),
        bad(
            whole[..4].to_vec(),
            Truncated,
            4,
            "the envelope version cut off",
        ),
        bad(whole[..6].to_vec(), Truncated, 6, "the schema id cut off"),
        bad(
            whole[..8].to_vec(),
            Truncated,
            8,
            "the schema version cut off",
        ),
        bad(whole[..10].to_vec(), Truncated, 10, "reserved cut off"),
        bad(
            whole[..12].to_vec(),
            Truncated,
            12,
            "the payload length cut off",
        ),
        bad(
            whole[..19].to_vec(),
            Truncated,
            12,
            "the payload length short by a byte",
        ),
        bad(with(&|f| f.magic = *b"TSRB"), Magic, 0, "another magic"),
        bad(
            with(&|f| f.envelope_version = 2),
            Version,
            4,
            "envelope version 2",
        ),
        bad(
            with(&|f| f.envelope_version = 0),
            Version,
            4,
            "envelope version 0",
        ),
        bad(
            with(&|f| f.schema = schema::TOKEN_FILE.id.0),
            Magic,
            6,
            "another schema of the catalog",
        ),
        bad(with(&|f| f.version = 1), Version, 8, "version v - 2"),
        bad(
            with(&|f| f.version = 4),
            Version,
            8,
            "a version newer than the reader's",
        ),
        bad(with(&|f| f.reserved = 1), Noncanonical, 10, "reserved is 1"),
        bad(
            with(&|f| f.reserved = 0x8000),
            Noncanonical,
            10,
            "reserved has its top bit set",
        ),
        bad(
            with(&|f| f.payload_len = Some(u64::MAX)),
            Limit,
            12,
            "a payload length whose envelope overflows a u64",
        ),
        bad(
            over.bytes(),
            Limit,
            12,
            "one byte over max_total_bytes in all",
        ),
        // The rest of the envelope.
        bad(
            whole[..whole.len() - DIGEST_LEN - 1].to_vec(),
            Truncated,
            12,
            "the payload short by a byte, and no digest",
        ),
        bad(
            whole[..whole.len() - 1].to_vec(),
            Truncated,
            12,
            "the digest short by a byte",
        ),
        bad(
            [&whole[..], &[0]].concat(),
            Noncanonical,
            end,
            "a byte after the digest",
        ),
        // The digest.
        bad(
            flipped(whole.clone(), whole.len() - 1),
            Digest,
            digest_at,
            "a digest bit flipped",
        ),
        bad(
            flipped(whole.clone(), HEADER_LEN + 8),
            Digest,
            digest_at,
            "a bit of an element flipped: the digest covers the payload",
        ),
        bad(
            {
                let mut bytes = whole.clone();
                bytes[8] = (PROBE_SCHEMA.version - 1) as u8;
                bytes
            },
            Digest,
            digest_at,
            "another accepted version under the same digest: the digest covers the header",
        ),
        // The payload, under a valid digest.
        bad(
            Fields::of([le(&[30], 8), vec![0; 16]].concat()).bytes(),
            Truncated,
            HEADER_LEN as u64,
            "30 elements declared, room for 2 (INV-TCE-3)",
        ),
        bad(
            undecodable.bytes(),
            Noncanonical,
            (HEADER_LEN + 16) as u64,
            "a byte left over in the payload (INV-TCE-4)",
        ),
        // Two faults: the check that comes first decides. A field is read before it is checked,
        // and each field is checked before the next is read.
        bad(
            b"TSX".to_vec(),
            Truncated,
            0,
            "a magic that cannot match, cut short: the field is incomplete before it is wrong",
        ),
        bad(
            [&b"TSRB"[..], &[1, 0]].concat(),
            Magic,
            0,
            "another magic, then the header cut short",
        ),
        bad(
            [&MAGIC[..], &[2, 0, 0xF0, 0xFF]].concat(),
            Version,
            4,
            "envelope version 2, then the header cut short",
        ),
        bad(
            with(&|f| {
                f.magic = *b"TSRB";
                f.envelope_version = 2;
            }),
            Magic,
            0,
            "another magic and envelope version 2",
        ),
        bad(
            with(&|f| {
                f.envelope_version = 2;
                f.schema = schema::TOKEN_FILE.id.0;
            }),
            Version,
            4,
            "envelope version 2 and another schema",
        ),
        bad(
            with(&|f| {
                f.schema = schema::TOKEN_FILE.id.0;
                f.version = 9;
                f.reserved = 1;
            }),
            Magic,
            6,
            "another schema, version 9 and reserved 1",
        ),
        bad(
            with(&|f| {
                f.version = 9;
                f.reserved = 1;
            }),
            Version,
            8,
            "version 9 and reserved 1",
        ),
        bad(
            with(&|f| {
                f.reserved = 1;
                f.payload_len = Some(u64::MAX);
            }),
            Noncanonical,
            10,
            "reserved 1 and a payload length over the limit",
        ),
        bad(
            flipped(with(&|f| f.reserved = 1), whole.len() - 1),
            Noncanonical,
            10,
            "reserved 1 and a bad digest: the header comes before the digest",
        ),
        bad(
            [&flipped(whole.clone(), whole.len() - 1)[..], &[0]].concat(),
            Noncanonical,
            end,
            "a byte after the digest and a bad digest: the length comes before the digest",
        ),
        bad(
            flipped(undecodable.bytes(), undecodable.bytes().len() - 1),
            Digest,
            (HEADER_LEN + undecodable.payload.len()) as u64,
            "an undecodable payload and a bad digest: the digest comes before the payload",
        ),
        bad(
            Fields::of([le(&[1 << 63], 8), vec![0; 16]].concat()).bytes(),
            Limit,
            HEADER_LEN as u64,
            "a 2^63 length with 16 bytes left, over max_vec_len and the bytes left: \
             max_vec_len comes first (INV-TCE-3)",
        ),
    ];
    // A reader of version 1, where v - 1 would be 0, which no writer emits.
    for (version, why) in [
        (0, "version 0 to a reader of version 1"),
        (2, "version 2 to a reader of version 1"),
    ] {
        let fields = Fields {
            schema: schema::CLAIM_LIST.id.0,
            version,
            ..base.clone()
        };
        cases.push(EnvelopeCase {
            expect: schema::CLAIM_LIST,
            ..bad(fields.bytes(), Version, 8, why)
        });
    }
    // RFC-0003's defaults, exactly at and one past each bound. A header alone shows the total:
    // within it, the reader goes on to find the payload missing.
    let header_only = |schema: Schema, payload_len: u64| {
        Fields {
            schema: schema.id.0,
            version: schema.version,
            payload_len: Some(payload_len),
            ..base.clone()
        }
        .header()
    };
    let overhead = (HEADER_LEN + DIGEST_LEN) as u64;
    let probe_total = |total: u64, outcome, why| {
        case(
            CaseLimits::Message,
            header_only(PROBE_SCHEMA, total - overhead),
            "Vec<u64>",
            outcome,
            why,
        )
    };
    cases.push(probe_total(
        RFC_MESSAGE_TOTAL,
        Err((Truncated, 12)),
        "exactly the 8 GiB message limit in all: the payload is missing",
    ));
    cases.push(probe_total(
        RFC_MESSAGE_TOTAL + 1,
        Err((Limit, 12)),
        "one byte over the 8 GiB message limit",
    ));
    for (total, outcome, why) in [
        (
            RFC_INTEGER_MODEL_TOTAL,
            Err((Truncated, 12)),
            "exactly the 1 TiB IntegerModel limit in all: the payload is missing",
        ),
        (
            RFC_INTEGER_MODEL_TOTAL + 1,
            Err((Limit, 12)),
            "one byte over the 1 TiB IntegerModel limit",
        ),
    ] {
        cases.push(EnvelopeCase {
            expect: schema::INTEGER_MODEL,
            limits: CaseLimits::IntegerModel,
            ty: "Vec<u64>",
            bytes: header_only(schema::INTEGER_MODEL, total - overhead),
            outcome,
            why,
        });
    }
    let message = |payload: Vec<u8>, ty, outcome, why| {
        case(
            CaseLimits::Message,
            Fields::of(payload).bytes(),
            ty,
            outcome,
            why,
        )
    };
    let most = RFC_MESSAGE_VEC_LEN;
    cases.push(message(
        le(&[most], 8),
        "Vec<u64>",
        Err((Truncated, HEADER_LEN as u64)),
        "2^34 elements, the most a message may declare, and none there",
    ));
    cases.push(message(
        le(&[most + 1], 8),
        "Vec<u64>",
        Err((Limit, HEADER_LEN as u64)),
        "2^34 + 1 elements, over the message limit",
    ));
    let depth = usize::from(RFC_MESSAGE_DEPTH);
    cases.push(message(
        somes(depth, &[7]),
        "Option^32<u8>",
        Ok(somes(depth, &[7])),
        "32 levels of nesting, the message limit",
    ));
    // The 33rd `Option` fails as it is entered, after 32 tags.
    let at = (HEADER_LEN + 2 * depth) as u64;
    cases.push(message(
        somes(depth + 1, &[7]),
        "Option^33<u8>",
        Err((Limit, at)),
        "33 levels of nesting, one over the message limit",
    ));
    cases
}
