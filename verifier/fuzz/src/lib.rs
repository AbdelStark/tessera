// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Decoders shared by the fuzz targets, built from `tv-codec`'s combinators.
//!
//! Each one returns the canonical encoding of the value it decoded, written from RFC-0003's
//! encoding table, so a target can check that an accepted input is exactly that encoding
//! (INV-TCE-2): a decoder that accepts a second encoding of some value fails the check.

use tv_codec::{Cursor, DecodeError, size};

/// The canonical bytes of a decoded value.
pub type Echo = Result<Vec<u8>, DecodeError>;

/// The types the targets decode, by selector: every type of the conformance vectors, plus a
/// few compositions they do not hold.
pub const TYPES: [&str; 36] = [
    "u8",
    "u16",
    "u32",
    "u64",
    "i8",
    "i16",
    "i32",
    "i64",
    "bool",
    "Fp",
    "Fp2",
    "Digest",
    "Bytes",
    "Ascii",
    "Rational",
    "Vec<u8>",
    "Vec<u16>",
    "Vec<u64>",
    "Vec<Fp>",
    "Vec<Fp2>",
    "Vec<Vec<i16>>",
    "Vec<Option<u64>>",
    "Option<u64>",
    "Option<Rational>",
    "Option^8<u8>",
    "[u16; 3]",
    "[u8; 1]^8",
    "pair^8",
    "(u8, Bytes)",
    "Map<u16, u8>",
    "Map<Bytes, u8>",
    "Map<u8, Map<u8, Ascii>>",
    "Sample",
    "Shape",
    "Vec<Shape>",
    "Option<[Fp2; 2]>",
];

/// Decodes one value of `TYPES[selector % TYPES.len()]`.
pub fn decode_as(selector: usize, c: &mut Cursor<'_>) -> Echo {
    match selector % TYPES.len() {
        0 => u8_(c),
        1 => u16_(c),
        2 => u32_(c),
        3 => u64_(c),
        4 => i8_(c),
        5 => i16_(c),
        6 => i32_(c),
        7 => i64_(c),
        8 => bool_(c),
        9 => fp(c),
        10 => fp2(c),
        11 => digest(c),
        12 => bytes(c),
        13 => ascii(c),
        14 => rational(c),
        15 => seq(c, size::U8, u8_),
        16 => seq(c, size::U16, u16_),
        17 => seq(c, size::U64, u64_),
        18 => seq(c, size::FP, fp),
        19 => seq(c, size::FP2, fp2),
        20 => seq(c, size::SEQUENCE, |c| seq(c, size::I16, i16_)),
        21 => seq(c, size::OPTION, |c| option(c, u64_)),
        22 => option(c, u64_),
        23 => option(c, rational),
        24 => options(c, 8),
        25 => array(c, 3, size::U16, u16_),
        26 => arrays(c, 8),
        27 => pairs(c, 8),
        28 => tuple(c, &[u8_, bytes]),
        29 => map(c, size::U16, size::U8, u16_, u8_),
        30 => map(c, size::SEQUENCE, size::U8, bytes, u8_),
        31 => map(c, size::U8, size::SEQUENCE, u8_, |c| {
            map(c, size::U8, size::SEQUENCE, u8_, ascii)
        }),
        32 => sample(c),
        33 => shape(c),
        34 => seq(c, size::TAG, shape),
        _ => option(c, |c| array(c, 2, size::FP2, fp2)),
    }
}

fn u64_le(value: u64) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

fn length(len: usize) -> Vec<u8> {
    u64_le(len as u64)
}

pub fn u8_(c: &mut Cursor<'_>) -> Echo {
    Ok(vec![c.u8()?])
}

pub fn u16_(c: &mut Cursor<'_>) -> Echo {
    Ok(c.u16()?.to_le_bytes().to_vec())
}

pub fn u32_(c: &mut Cursor<'_>) -> Echo {
    Ok(c.u32()?.to_le_bytes().to_vec())
}

pub fn u64_(c: &mut Cursor<'_>) -> Echo {
    Ok(u64_le(c.u64()?))
}

pub fn i8_(c: &mut Cursor<'_>) -> Echo {
    Ok(c.i8()?.to_le_bytes().to_vec())
}

pub fn i16_(c: &mut Cursor<'_>) -> Echo {
    Ok(c.i16()?.to_le_bytes().to_vec())
}

pub fn i32_(c: &mut Cursor<'_>) -> Echo {
    Ok(c.i32()?.to_le_bytes().to_vec())
}

pub fn i64_(c: &mut Cursor<'_>) -> Echo {
    Ok(c.i64()?.to_le_bytes().to_vec())
}

pub fn bool_(c: &mut Cursor<'_>) -> Echo {
    Ok(vec![u8::from(c.bool()?)])
}

pub fn fp(c: &mut Cursor<'_>) -> Echo {
    Ok(u64_le(c.fp()?.to_canonical()))
}

pub fn fp2(c: &mut Cursor<'_>) -> Echo {
    let x = c.fp2()?;
    Ok([u64_le(x.c0.to_canonical()), u64_le(x.c1.to_canonical())].concat())
}

pub fn digest(c: &mut Cursor<'_>) -> Echo {
    Ok(c.digest()?.to_vec())
}

pub fn bytes(c: &mut Cursor<'_>) -> Echo {
    let b = c.bytes()?;
    Ok([length(b.len()), b.to_vec()].concat())
}

pub fn ascii(c: &mut Cursor<'_>) -> Echo {
    let text = c.ascii()?;
    Ok([length(text.len()), text.as_bytes().to_vec()].concat())
}

pub fn rational(c: &mut Cursor<'_>) -> Echo {
    let r = c.rational()?;
    Ok([r.num().to_le_bytes(), r.den().to_le_bytes()].concat())
}

/// `Vec<T>`.
pub fn seq(
    c: &mut Cursor<'_>,
    element_min: u64,
    element: impl FnMut(&mut Cursor<'_>) -> Echo,
) -> Echo {
    let items = c.vec(element_min, element)?;
    Ok([length(items.len()), items.concat()].concat())
}

/// `Option<T>`.
pub fn option(c: &mut Cursor<'_>, some: impl FnOnce(&mut Cursor<'_>) -> Echo) -> Echo {
    Ok(match c.option(some)? {
        None => vec![0, 0],
        Some(value) => [vec![1, 0], value].concat(),
    })
}

/// `[T; len]`.
pub fn array(
    c: &mut Cursor<'_>,
    len: usize,
    element_min: u64,
    element: impl FnMut(&mut Cursor<'_>) -> Echo,
) -> Echo {
    Ok(c.array(len, element_min, element)?.concat())
}

/// A tuple of these fields.
pub fn tuple(c: &mut Cursor<'_>, fields: &[fn(&mut Cursor<'_>) -> Echo]) -> Echo {
    c.nested(|c| {
        let mut out = Vec::new();
        for field in fields {
            out.extend(field(c)?);
        }
        Ok(out)
    })
}

/// `Map<K, V>`.
pub fn map(
    c: &mut Cursor<'_>,
    key_min: u64,
    value_min: u64,
    key: impl FnMut(&mut Cursor<'_>) -> Echo,
    value: impl FnMut(&mut Cursor<'_>) -> Echo,
) -> Echo {
    let entries = c.map(key_min, value_min, key, value)?;
    let mut out = length(entries.len());
    for (k, v) in entries {
        out.extend(k);
        out.extend(v);
    }
    Ok(out)
}

/// `Option^n<u8>`.
pub fn options(c: &mut Cursor<'_>, n: usize) -> Echo {
    match n {
        0 => u8_(c),
        _ => option(c, |c| options(c, n - 1)),
    }
}

/// `[u8; 1]^n`.
pub fn arrays(c: &mut Cursor<'_>, n: usize) -> Echo {
    match n {
        0 => u8_(c),
        _ => array(c, 1, 1, |c| arrays(c, n - 1)),
    }
}

/// `pair^n`: `n` nested pairs, each adding a `u8` on the right.
pub fn pairs(c: &mut Cursor<'_>, n: usize) -> Echo {
    match n {
        0 => u8_(c),
        _ => c.nested(|c| Ok([pairs(c, n - 1)?, u8_(c)?].concat())),
    }
}

/// `struct Sample { flag: bool, small: u8, list: Vec<u16>, maybe: Option<i32>, name: Ascii,
/// digest: [u8; 32] }`, the test schema of `spec/vectors/codec/`.
pub fn sample(c: &mut Cursor<'_>) -> Echo {
    c.nested(|c| {
        Ok([
            bool_(c)?,
            u8_(c)?,
            seq(c, size::U16, u16_)?,
            option(c, i32_)?,
            ascii(c)?,
            digest(c)?,
        ]
        .concat())
    })
}

/// `enum Shape { Empty, Point(u32), Box { width: u16, height: u16 }, Nested(Vec<Shape>) }`.
pub fn shape(c: &mut Cursor<'_>) -> Echo {
    c.enumeration(4, |c, tag| {
        let fields = match tag {
            0 => Vec::new(),
            1 => u32_(c)?,
            2 => [u16_(c)?, u16_(c)?].concat(),
            _ => seq(c, size::TAG, shape)?,
        };
        Ok([tag.to_le_bytes().to_vec(), fields].concat())
    })
}
