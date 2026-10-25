// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Strict Ed25519 verification of recorder manifests (RFC-0015 'Signatures', RFC-0060
//! 'Signatures'), written from RFC 8032 §5.1 and RFC-0060 in safe Rust, with no third-party
//! code: the field, the curve, the scalar reduction and SHA-512 ([`sha512`](crate::sha512)) are
//! this crate's own.
//!
//! RFC-0060 defines the acceptance set as that of `ed25519-dalek` 2.2's `verify_strict` without
//! `legacy_compatibility`, the prover's implementation; [`verify_strict`] implements the same set
//! from RFC-0060's statement of it. Pure Ed25519 without a context: the message is hashed as it
//! is given.
//!
//! # Specification
//!
//! [`verify_strict`]`(A, M, R ‖ S)` accepts exactly when every rule below holds, and otherwise
//! returns the first that fails, in this order, as [`Ed25519Error`]:
//!
//! 1. `S`, the last 32 bytes as a little-endian integer of all 256 bits, is below
//!    `L = 2^252 + 27742317777372353535851937790883648493`, else `ScalarOutOfRange`.
//!    RFC-0060: "`S < L`".
//! 2. `R` decodes to a point, else `RNotAPoint`; 3. `A` decodes to a point, else `KeyNotAPoint`.
//!    Decoding is RFC 8032 §5.1.3 except that a `y` in `[p, 2^255)` is reduced modulo `p`
//!    instead of refused. RFC-0060: "`R` and `A` decode to points ... `curve25519-dalek` 4
//!    decodes as RFC 8032 §5.1.3 does, except that it also accepts the non-canonical encodings
//!    `y ∈ [p, 2^255)`, which that section refuses". The rest of §5.1.3 holds, so an encoding
//!    with `x = 0` and bit 255 set does not decode; the two points with `x = 0` are of small
//!    order, refused by rule 4 or 5 had they decoded, so the acceptance set is the same either
//!    way.
//! 4. `R` is not of small order, else `RSmallOrder`; 5. `A` is not of small order, else
//!    `KeySmallOrder`. A point is of small order when `[8]P` is the identity: the eight points
//!    of the torsion subgroup. A point of mixed order, a torsion point added to a point of order
//!    `L`, is not of small order and passes. RFC-0060: "neither is of small order".
//! 6. `R`'s 32 bytes equal the canonical encoding (RFC 8032 §5.1.2) of `[S]B − [k]A`, with
//!    `k = SHA-512(R ‖ A ‖ M)` as a little-endian integer reduced modulo `L`, `R` and `A` hashed
//!    as the 32 bytes given; else `Equation`. RFC-0060: "`R`'s 32 bytes equal the encoding of
//!    `[S]B − [k]A`, with `k = SHA-512(R ‖ A ‖ message)` reduced modulo `L`. That is the
//!    cofactorless equation, and it leaves `R` only its canonical encoding."
//!
//! Consequences the tests pin down: a non-canonical `R` decodes but fails rule 6, since an
//! encoding computed by rule 6 is canonical; a non-canonical `A` decodes, and `k` hashes its
//! bytes as given; `S + L` fails rule 1 although the equation would hold modulo `L`; a
//! mixed-order key whose torsion component `[k]` annihilates passes.
//!
//! Every input yields `Ok(())` or one of these errors; nothing panics and nothing allocates. The
//! work is fixed by the input sizes: two decodings, two small-order checks, one SHA-512 over
//! `64 + |M|` bytes and two scalar multiplications. Nothing is constant-time, and nothing needs
//! to be: keys, messages and signatures are all public.
//!
//! ```
//! use tv_hash::ed25519::{Ed25519Error, verify_strict};
//!
//! // RFC 8032 §7.1, TEST 1: the empty message.
//! let hex = |s: &str| -> Vec<u8> {
//!     (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
//! };
//! let key: [u8; 32] = hex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
//!     .try_into()
//!     .unwrap();
//! let mut signature: [u8; 64] = hex(concat!(
//!     "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155",
//!     "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
//! ))
//! .try_into()
//! .unwrap();
//! assert_eq!(verify_strict(&key, b"", &signature), Ok(()));
//! assert_eq!(verify_strict(&key, b"x", &signature), Err(Ed25519Error::Equation));
//! signature[63] |= 0x80;
//! assert_eq!(verify_strict(&key, b"", &signature), Err(Ed25519Error::ScalarOutOfRange));
//! ```

mod field;
mod point;
mod scalar;

use core::fmt;

use crate::sha512::Sha512;
use point::Point;

/// The length of a public key, the encoding of `A`.
pub const PUBLIC_KEY_LEN: usize = 32;

/// The length of a signature, `R ‖ S`.
pub const SIGNATURE_LEN: usize = 64;

/// The first rule of strict verification that a signature fails, in the order of
/// [`verify_strict`]. Every field is public; none carries data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ed25519Error {
    /// Rule 1: `S ≥ L`.
    ScalarOutOfRange,
    /// Rule 2: `R` decodes to no point.
    RNotAPoint,
    /// Rule 3: the public key `A` decodes to no point.
    KeyNotAPoint,
    /// Rule 4: `R` is of small order.
    RSmallOrder,
    /// Rule 5: the public key `A` is of small order.
    KeySmallOrder,
    /// Rule 6: the encoding of `[S]B − [k]A` is not `R`'s bytes.
    Equation,
}

impl fmt::Display for Ed25519Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Ed25519Error::ScalarOutOfRange => "the signature's S is not below the group order",
            Ed25519Error::RNotAPoint => "the signature's R is not the encoding of a point",
            Ed25519Error::KeyNotAPoint => "the public key is not the encoding of a point",
            Ed25519Error::RSmallOrder => "the signature's R is a point of small order",
            Ed25519Error::KeySmallOrder => "the public key is a point of small order",
            Ed25519Error::Equation => "the signature does not satisfy [S]B = R + [k]A",
        })
    }
}

impl std::error::Error for Ed25519Error {}

/// Verifies `signature`, `R ‖ S`, on `message` under `public_key`, `A`, by the rules of the
/// [module documentation](self), in their order.
pub fn verify_strict(
    public_key: &[u8; PUBLIC_KEY_LEN],
    message: &[u8],
    signature: &[u8; SIGNATURE_LEN],
) -> Result<(), Ed25519Error> {
    let mut r_bytes = [0u8; 32];
    let mut s_bytes = [0u8; 32];
    for (slot, byte) in r_bytes.iter_mut().zip(signature) {
        *slot = *byte;
    }
    for (slot, byte) in s_bytes.iter_mut().zip(signature.iter().skip(32)) {
        *slot = *byte;
    }

    // 1.
    if !scalar::is_below_l(&s_bytes) {
        return Err(Ed25519Error::ScalarOutOfRange);
    }
    // 2, 3.
    let r = Point::decode(&r_bytes).ok_or(Ed25519Error::RNotAPoint)?;
    let a = Point::decode(public_key).ok_or(Ed25519Error::KeyNotAPoint)?;
    // 4, 5.
    if r.is_small_order() {
        return Err(Ed25519Error::RSmallOrder);
    }
    if a.is_small_order() {
        return Err(Ed25519Error::KeySmallOrder);
    }
    // 6.
    let digest = Sha512::new()
        .update(&r_bytes)
        .update(public_key)
        .update(message)
        .finalize();
    let k = scalar::reduce_wide(&digest);
    let expected = Point::BASE
        .multiple(&s_bytes)
        .plus(&a.negate().multiple(&k));
    if expected.encode() == r_bytes {
        Ok(())
    } else {
        Err(Ed25519Error::Equation)
    }
}
