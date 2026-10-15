// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The field cases of `spec/vectors/codec/` (see `tessera_codec::golden`): the encodings of
//! `Fp` and `Fp2` with their values, and the non-canonical values their decoders must reject.

use tessera_codec::error::ErrorCode;
use tessera_codec::golden::{Boundary, Golden, Outcome, Rejected, Show, Shown};
use tessera_codec::{Decode, DecodeLimits, Encode, decode, encode};

use crate::{Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};

const P: u64 = Goldilocks::MODULUS;

/// An element as its canonical value.
impl Show for Goldilocks {
    fn show(&self) -> Shown {
        Shown::Int(self.to_canonical_u64().to_string())
    }
}

/// `[c0, c1]`.
impl Show for GoldilocksExt2 {
    fn show(&self) -> Shown {
        let [c0, c1] = self.coeffs();
        Shown::List(vec![c0.show(), c1.show()])
    }
}

fn fp(x: u64) -> Goldilocks {
    Goldilocks::from_u64_reduced(x)
}

/// The golden cases of `Fp` and `Fp2`.
pub fn cases() -> Vec<Golden> {
    vec![
        Golden::of("Fp", fp(0)),
        Golden::of("Fp", fp(1)),
        Golden::of("Fp", fp(P - 1)),
        Golden::of("Fp", fp(1 << 32)),
        Golden::of("Fp", fp(0x0102_0304_0506_0708)),
        Golden::of("Fp2", GoldilocksExt2::new(fp(0), fp(0))),
        Golden::of("Fp2", GoldilocksExt2::new(fp(1), fp(P - 1))),
        Golden::of("Fp2", GoldilocksExt2::new(fp(7), fp(1 << 63))),
        Golden::of("Vec<Fp>", vec![fp(2), fp(P - 2)]),
    ]
}

/// The non-canonical corpus of `Fp` and `Fp2`.
pub fn corpus() -> Vec<Rejected> {
    use ErrorCode::{CodecNoncanonical as Noncanonical, CodecTruncated as Truncated};
    let words =
        |values: &[u64]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let bad = |ty, bytes, code, why| Rejected {
        ty,
        bytes,
        code,
        why,
    };
    vec![
        bad("Fp", words(&[P]), Noncanonical, "the modulus itself"),
        bad(
            "Fp",
            words(&[P + 1]),
            Noncanonical,
            "a value above the modulus",
        ),
        bad("Fp", words(&[u64::MAX]), Noncanonical, "2^64 − 1"),
        bad("Fp", vec![1; 7], Truncated, "seven bytes of eight"),
        bad("Fp2", words(&[P, 0]), Noncanonical, "a non-canonical c0"),
        bad("Fp2", words(&[0, P]), Noncanonical, "a non-canonical c1"),
        bad("Fp2", words(&[1]), Truncated, "c0 without c1"),
        bad(
            "Vec<Fp>",
            [words(&[2]), words(&[3, u64::MAX])].concat(),
            Noncanonical,
            "a non-canonical element",
        ),
        bad(
            "Option^8<Fp2>",
            [[1u8, 0].repeat(8), words(&[1, 2])].concat(),
            ErrorCode::CodecLimit,
            "eight Options around an Fp2, which is one level",
        ),
    ]
}

/// The field inputs exactly at the depth limit, which decoders must accept.
pub fn boundaries() -> Vec<Boundary> {
    vec![Boundary {
        ty: "Option^7<Fp2>",
        bytes: [
            [1u8, 0].repeat(7),
            1u64.to_le_bytes().to_vec(),
            2u64.to_le_bytes().to_vec(),
        ]
        .concat(),
        why: "seven Options and an Fp2: depth 8",
    }]
}

/// Decodes `bytes` as the field type named `ty`; `None` for other types.
pub fn decode_as(ty: &str, bytes: &[u8], limits: DecodeLimits) -> Outcome {
    fn read<T: Decode + Encode + Show>(bytes: &[u8], limits: DecodeLimits) -> Outcome {
        Some(decode::<T>(bytes, limits).map(|value| (value.show(), encode(&value))))
    }
    match ty {
        "Fp" => read::<Goldilocks>(bytes, limits),
        "Fp2" => read::<GoldilocksExt2>(bytes, limits),
        "Vec<Fp>" => read::<Vec<Goldilocks>>(bytes, limits),
        "Option^7<Fp2>" => read::<
            Option<Option<Option<Option<Option<Option<Option<GoldilocksExt2>>>>>>>,
        >(bytes, limits),
        "Option^8<Fp2>" => read::<
            Option<Option<Option<Option<Option<Option<Option<Option<GoldilocksExt2>>>>>>>>,
        >(bytes, limits),
        _ => None,
    }
}
