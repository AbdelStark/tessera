// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The quadratic extension `Fp2 = Fp[X]/(X^2 − W)` (RFC-0002 'Quadratic extension') over
//! Goldilocks (`W = 7`) and the test field (`W = 3`): the oracle vectors of
//! `spec/vectors/field/goldilocks_ext2.json`, the schoolbook product, the norm, inverses and the
//! non-residue.

use std::fs;
use std::path::Path;

use proptest::prelude::*;
use serde_json::Value;
use tessera_field::{
    ExtensionBase, F65537, F65537Ext2, Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExt,
    QuadraticExtension,
};

fn base<B: PrimeField>() -> impl Strategy<Value = B> {
    prop_oneof![
        1 => proptest::sample::select(vec![B::ZERO, B::ONE, -B::ONE, B::from_u64_reduced(1 << 32)]),
        4 => any::<u64>().prop_map(B::from_u64_reduced),
    ]
}

fn ext<B: ExtensionBase>() -> impl Strategy<Value = QuadraticExt<B>> {
    (base::<B>(), base::<B>()).prop_map(|(c0, c1)| QuadraticExt::new(c0, c1))
}

/// The product of `X^2 − W` written out without Karatsuba.
fn schoolbook<B: ExtensionBase>(a: QuadraticExt<B>, b: QuadraticExt<B>) -> QuadraticExt<B> {
    let ([a0, a1], [b0, b1]) = (a.coeffs(), b.coeffs());
    let w = <QuadraticExt<B> as QuadraticExtension<B>>::NON_RESIDUE;
    QuadraticExt::new(a0 * b0 + w * a1 * b1, a0 * b1 + a1 * b0)
}

fn extension_laws<B: ExtensionBase>(a: QuadraticExt<B>, b: QuadraticExt<B>, c: QuadraticExt<B>) {
    assert_eq!(a * b, schoolbook(a, b));
    assert_eq!((a * b) * c, a * (b * c));
    assert_eq!(a * (b + c), a * b + a * c);
    assert_eq!(a.square(), a * a);
    assert_eq!(a * QuadraticExt::ONE, a);
    assert_eq!(a + (-a), QuadraticExt::ZERO);
    // The norm is multiplicative and equals a · conjugate(a), embedded in the extension.
    assert_eq!((a * b).norm(), a.norm() * b.norm());
    assert_eq!(a * a.conjugate(), QuadraticExt::from(a.norm()));
    match a.inverse() {
        Some(inverse) => assert_eq!(a * inverse, QuadraticExt::ONE),
        None => assert_eq!(a, QuadraticExt::ZERO),
    }
    let [b0, _] = b.coeffs();
    assert_eq!(a.mul_base(b0), a * QuadraticExt::from(b0));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1 << 16))]

    #[test]
    fn goldilocks_ext2_is_a_field_extension(a in ext::<Goldilocks>(), b in ext::<Goldilocks>(), c in ext::<Goldilocks>()) {
        extension_laws(a, b, c);
    }

    #[test]
    fn f65537_ext2_is_a_field_extension(a in ext::<F65537>(), b in ext::<F65537>(), c in ext::<F65537>()) {
        extension_laws(a, b, c);
    }

    #[test]
    fn small_multiplication_matches_full_multiplication(x in any::<u64>(), c in any::<u32>()) {
        let a = Goldilocks::from_u64_reduced(x);
        prop_assert_eq!(a.mul_small(c), a * Goldilocks::from_u64_reduced(u64::from(c)));
    }
}

/// `X^2 − W` is irreducible exactly when `W` is a non-residue: `W^((P − 1)/2) = −1`.
#[test]
fn the_non_residues_are_non_residues() {
    let w = <GoldilocksExt2 as QuadraticExtension<Goldilocks>>::NON_RESIDUE;
    assert_eq!(w, Goldilocks::from_u64_reduced(7));
    assert_eq!(w.pow((Goldilocks::MODULUS - 1) / 2), -Goldilocks::ONE);
    let w = <F65537Ext2 as QuadraticExtension<F65537>>::NON_RESIDUE;
    assert_eq!(w, F65537::from_u64_reduced(3));
    assert_eq!(w.pow((F65537::MODULUS - 1) / 2), -F65537::ONE);
    // Every smaller candidate is a residue, so 7 is the smallest non-residue (RFC-0002).
    for small in 2..7u64 {
        let x = Goldilocks::from_u64_reduced(small);
        assert_eq!(
            x.pow((Goldilocks::MODULUS - 1) / 2),
            Goldilocks::ONE,
            "{small}"
        );
    }
}

#[test]
fn small_multiplication_at_the_edges() {
    for x in [0, 1, Goldilocks::MODULUS - 1, (1 << 32) - 1, 1 << 63] {
        let a = Goldilocks::from_u64_reduced(x);
        for c in [0, 1, 7, u32::MAX] {
            assert_eq!(
                a.mul_small(c),
                a * Goldilocks::from_u64_reduced(u64::from(c)),
                "{x} * {c}"
            );
        }
    }
}

fn hex(value: &Value) -> Goldilocks {
    let text = value.as_str().expect("values are hex strings");
    let x = u64::from_str_radix(text.trim_start_matches("0x"), 16).expect("valid hex");
    Goldilocks::from_canonical_u64(x).expect("canonical")
}

#[test]
fn ext2_arithmetic_matches_the_oracle() {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/field/goldilocks_ext2.json");
    let file: Value =
        serde_json::from_str(&fs::read_to_string(path).expect("vector file")).expect("JSON");
    assert_eq!(file["non_residue"].as_u64(), Some(7));
    let cases = file["cases"].as_array().expect("cases");
    assert!(cases.len() >= 4096 + 49);
    for case in cases {
        let c = case.as_array().expect("case");
        let a = GoldilocksExt2::new(hex(&c[0]), hex(&c[1]));
        let b = GoldilocksExt2::new(hex(&c[2]), hex(&c[3]));
        assert_eq!(
            a * b,
            GoldilocksExt2::new(hex(&c[4]), hex(&c[5])),
            "{a:?} * {b:?}"
        );
        assert_eq!(a.norm(), hex(&c[8]), "norm {a:?}");
        match (&c[6], &c[7]) {
            (Value::Null, Value::Null) => assert_eq!(a.inverse(), None),
            (i0, i1) => assert_eq!(
                a.inverse(),
                Some(GoldilocksExt2::new(hex(i0), hex(i1))),
                "{a:?}^-1"
            ),
        }
    }
}
