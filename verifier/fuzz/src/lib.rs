// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Decoders shared by the fuzz targets, built from `tv-codec`'s combinators, and what each
//! reports about the value it decoded, for the targets' oracles. The cases and oracles of the
//! proof targets, `multiproof` and `inclusion`, are in [`proofs`], and those of `sumcheck` in
//! [`sumcheck`].
//!
//! Each decoder returns an [`Echo`]: the canonical encoding of its value, written from RFC-0003's
//! encoding table from the value and never copied from the input, with the value's nesting depth
//! and its longest sequence. A target compares the encoding with its input (INV-TCE-2) and the
//! depth and lengths with its limits, so a decoder that accepts a second encoding of a value, or
//! a value past a limit, fails the target. What no value of the type can be panics here: a
//! rational with a zero denominator, text outside `Ascii`, two equal map keys, an unknown enum
//! tag.

use tv_codec::{Cursor, DecodeError, size};

pub mod proofs;
pub mod sumcheck;

/// What a decoder learned about the value it decoded.
pub struct Echo {
    /// The value's canonical encoding.
    pub bytes: Vec<u8>,
    /// Its nesting depth, as RFC-0003 'Limits' counts levels.
    pub depth: u16,
    /// The length of its longest sequence: `Vec`, `Bytes`, `Ascii` or map.
    pub longest: u64,
}

/// A decoder's result.
pub type Decoded = Result<Echo, DecodeError>;

impl Echo {
    /// A value that holds no other: integers, `bool` and `Fp`.
    fn leaf(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            depth: 0,
            longest: 0,
        }
    }

    /// A value one level above the deepest of its `parts`, encoded as `head` followed by them; a
    /// sequence of `len` elements has a `len` length.
    fn node(head: Vec<u8>, parts: Vec<Echo>, len: u64) -> Self {
        let depth = parts.iter().map(|part| part.depth).max().unwrap_or(0);
        let longest = parts.iter().map(|part| part.longest).max().unwrap_or(0);
        let mut bytes = head;
        for part in parts {
            bytes.extend(part.bytes);
        }
        Self {
            bytes,
            depth: depth.saturating_add(1),
            longest: longest.max(len),
        }
    }
}

/// The types the targets decode, by selector: every type of the conformance vectors, plus a
/// few compositions they do not hold, and the artifacts whose decoders `tv-codec` exports.
pub const TYPES: [&str; 37] = [
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
    "WeightCommitmentPublic",
];

/// Decodes one value of `TYPES[selector % TYPES.len()]`.
pub fn decode_as(selector: usize, c: &mut Cursor<'_>) -> Decoded {
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
        35 => option(c, |c| array(c, 2, size::FP2, fp2)),
        _ => weight_commitment_public(c),
    }
}

fn length(len: usize) -> Vec<u8> {
    (len as u64).to_le_bytes().to_vec()
}

pub fn u8_(c: &mut Cursor<'_>) -> Decoded {
    Ok(Echo::leaf(vec![c.u8()?]))
}

pub fn u16_(c: &mut Cursor<'_>) -> Decoded {
    Ok(Echo::leaf(c.u16()?.to_le_bytes().to_vec()))
}

pub fn u32_(c: &mut Cursor<'_>) -> Decoded {
    Ok(Echo::leaf(c.u32()?.to_le_bytes().to_vec()))
}

pub fn u64_(c: &mut Cursor<'_>) -> Decoded {
    Ok(Echo::leaf(c.u64()?.to_le_bytes().to_vec()))
}

pub fn i8_(c: &mut Cursor<'_>) -> Decoded {
    Ok(Echo::leaf(c.i8()?.to_le_bytes().to_vec()))
}

pub fn i16_(c: &mut Cursor<'_>) -> Decoded {
    Ok(Echo::leaf(c.i16()?.to_le_bytes().to_vec()))
}

pub fn i32_(c: &mut Cursor<'_>) -> Decoded {
    Ok(Echo::leaf(c.i32()?.to_le_bytes().to_vec()))
}

pub fn i64_(c: &mut Cursor<'_>) -> Decoded {
    Ok(Echo::leaf(c.i64()?.to_le_bytes().to_vec()))
}

pub fn bool_(c: &mut Cursor<'_>) -> Decoded {
    Ok(Echo::leaf(vec![u8::from(c.bool()?)]))
}

pub fn fp(c: &mut Cursor<'_>) -> Decoded {
    Ok(Echo::leaf(c.fp()?.to_canonical().to_le_bytes().to_vec()))
}

pub fn fp2(c: &mut Cursor<'_>) -> Decoded {
    let x = c.fp2()?;
    let coefficients = vec![
        Echo::leaf(x.c0.to_canonical().to_le_bytes().to_vec()),
        Echo::leaf(x.c1.to_canonical().to_le_bytes().to_vec()),
    ];
    Ok(Echo::node(Vec::new(), coefficients, 0))
}

pub fn digest(c: &mut Cursor<'_>) -> Decoded {
    let bytes = c.digest()?.iter().map(|&b| Echo::leaf(vec![b])).collect();
    Ok(Echo::node(Vec::new(), bytes, 0))
}

pub fn bytes(c: &mut Cursor<'_>) -> Decoded {
    let b = c.bytes()?;
    Ok(Echo::node(
        [length(b.len()), b.to_vec()].concat(),
        Vec::new(),
        b.len() as u64,
    ))
}

pub fn ascii(c: &mut Cursor<'_>) -> Decoded {
    let text = c.ascii()?;
    assert!(
        text.bytes().all(|b| (0x20..=0x7E).contains(&b)),
        "Ascii accepted with a byte outside 0x20..=0x7E: {:02x?}",
        text.as_bytes()
    );
    Ok(Echo::node(
        [length(text.len()), text.as_bytes().to_vec()].concat(),
        Vec::new(),
        text.len() as u64,
    ))
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// `Rational`, encoded in lowest terms whatever its decoder returned.
pub fn rational(c: &mut Cursor<'_>) -> Decoded {
    let r = c.rational()?;
    let (num, den) = (r.num(), r.den());
    assert!(den != 0, "a rational accepted with denominator 0");
    let g = gcd(num.unsigned_abs(), den);
    let num = i64::try_from(i128::from(num) / i128::from(g)).expect("in range");
    Ok(Echo::node(
        [num.to_le_bytes(), (den / g).to_le_bytes()].concat(),
        Vec::new(),
        0,
    ))
}

/// `Vec<T>`.
pub fn seq(
    c: &mut Cursor<'_>,
    element_min: u64,
    element: impl FnMut(&mut Cursor<'_>) -> Decoded,
) -> Decoded {
    let items = c.vec(element_min, element)?;
    let len = items.len();
    Ok(Echo::node(length(len), items, len as u64))
}

/// `Option<T>`.
pub fn option(c: &mut Cursor<'_>, some: impl FnOnce(&mut Cursor<'_>) -> Decoded) -> Decoded {
    Ok(match c.option(some)? {
        None => Echo::node(vec![0, 0], Vec::new(), 0),
        Some(value) => Echo::node(vec![1, 0], vec![value], 0),
    })
}

/// `[T; len]`.
pub fn array(
    c: &mut Cursor<'_>,
    len: usize,
    element_min: u64,
    element: impl FnMut(&mut Cursor<'_>) -> Decoded,
) -> Decoded {
    Ok(Echo::node(
        Vec::new(),
        c.array(len, element_min, element)?,
        0,
    ))
}

/// A tuple of these fields.
pub fn tuple(c: &mut Cursor<'_>, fields: &[fn(&mut Cursor<'_>) -> Decoded]) -> Decoded {
    c.nested(|c| {
        let parts = fields
            .iter()
            .map(|field| field(c))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Echo::node(Vec::new(), parts, 0))
    })
}

/// `Map<K, V>`, encoded with its entries sorted by their keys' encodings whatever order its
/// decoder returned them in; two equal keys panic.
pub fn map(
    c: &mut Cursor<'_>,
    key_min: u64,
    value_min: u64,
    key: impl FnMut(&mut Cursor<'_>) -> Decoded,
    value: impl FnMut(&mut Cursor<'_>) -> Decoded,
) -> Decoded {
    let mut entries = c.map(key_min, value_min, key, value)?;
    entries.sort_by(|a, b| a.0.bytes.cmp(&b.0.bytes));
    assert!(
        entries
            .windows(2)
            .all(|pair| pair[0].0.bytes != pair[1].0.bytes),
        "a map accepted with two equal keys"
    );
    let len = entries.len();
    let entries = entries
        .into_iter()
        .map(|(k, v)| Echo::node(Vec::new(), vec![k, v], 0))
        .collect();
    Ok(Echo::node(length(len), entries, len as u64))
}

/// `Option^n<u8>`.
pub fn options(c: &mut Cursor<'_>, n: usize) -> Decoded {
    match n {
        0 => u8_(c),
        _ => option(c, |c| options(c, n - 1)),
    }
}

/// `[u8; 1]^n`.
pub fn arrays(c: &mut Cursor<'_>, n: usize) -> Decoded {
    match n {
        0 => u8_(c),
        _ => array(c, 1, 1, |c| arrays(c, n - 1)),
    }
}

/// `pair^n`: `n` nested pairs, each adding a `u8` on the right.
pub fn pairs(c: &mut Cursor<'_>, n: usize) -> Decoded {
    match n {
        0 => u8_(c),
        _ => c.nested(|c| Ok(Echo::node(Vec::new(), vec![pairs(c, n - 1)?, u8_(c)?], 0))),
    }
}

/// `struct Sample { flag: bool, small: u8, list: Vec<u16>, maybe: Option<i32>, name: Ascii,
/// digest: [u8; 32] }`, the test schema of `spec/vectors/codec/`.
pub fn sample(c: &mut Cursor<'_>) -> Decoded {
    c.nested(|c| {
        let fields = vec![
            bool_(c)?,
            u8_(c)?,
            seq(c, size::U16, u16_)?,
            option(c, i32_)?,
            ascii(c)?,
            digest(c)?,
        ];
        Ok(Echo::node(Vec::new(), fields, 0))
    })
}

/// `enum Shape { Empty, Point(u32), Box { width: u16, height: u16 }, Nested(Vec<Shape>) }`,
/// encoded by the variant decoded, whose fields take no level of their own.
pub fn shape(c: &mut Cursor<'_>) -> Decoded {
    c.enumeration(4, |c, tag| {
        let (variant, fields): (u16, Vec<Echo>) = match tag {
            0 => (0, Vec::new()),
            1 => (1, vec![u32_(c)?]),
            2 => (2, vec![u16_(c)?, u16_(c)?]),
            3 => (3, vec![seq(c, size::TAG, shape)?]),
            other => panic!("tag {other} accepted for an enum of 4 variants"),
        };
        Ok(Echo::node(variant.to_le_bytes().to_vec(), fields, 0))
    })
}

/// `WeightCommitmentPublic`, schema `0x0301` (RFC-0040), decoded by `tv-codec`'s decoder and
/// encoded here from RFC-0040's field list, never by the type's own `encode`: a struct of
/// `layout`, itself a struct of `log2_k: u8`, `log2_inv_rate: u8`, `mask_coeffs: u32` and
/// `blinding_rows: u32`; `roots: [Digest; 10]`; `epoch: u32`. Three levels deep: the struct,
/// then the layout and the roots, then each root.
pub fn weight_commitment_public(c: &mut Cursor<'_>) -> Decoded {
    let value = tv_codec::WeightCommitmentPublic::decode(c)?;
    let layout = Echo::node(
        Vec::new(),
        vec![
            Echo::leaf(vec![value.layout.log2_k]),
            Echo::leaf(vec![value.layout.log2_inv_rate]),
            Echo::leaf(value.layout.mask_coeffs.to_le_bytes().to_vec()),
            Echo::leaf(value.layout.blinding_rows.to_le_bytes().to_vec()),
        ],
        0,
    );
    let roots = value
        .roots
        .iter()
        .map(|root| {
            Echo::node(
                Vec::new(),
                root.iter().map(|&b| Echo::leaf(vec![b])).collect(),
                0,
            )
        })
        .collect();
    Ok(Echo::node(
        Vec::new(),
        vec![
            layout,
            Echo::node(Vec::new(), roots, 0),
            Echo::leaf(value.epoch.to_le_bytes().to_vec()),
        ],
        0,
    ))
}
