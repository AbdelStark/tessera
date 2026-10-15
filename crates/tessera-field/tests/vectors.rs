// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The field against the independent oracle: `spec/vectors/field/`, produced by
//! `spec/vectors/field/generate.py` with Python's arbitrary-precision integers (RFC-0002
//! 'Testing Strategy'). The verifier's `tv-field` checks the same files.

use std::fs;
use std::path::Path;

use serde_json::Value;
use tessera_field::{F65537, Field, Goldilocks, PrimeField};

fn vectors(name: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../spec/vectors/field")
        .join(name);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).expect("vector files are JSON")
}

fn hex(value: &Value) -> u64 {
    let text = value.as_str().expect("values are hex strings");
    u64::from_str_radix(text.trim_start_matches("0x"), 16).expect("valid hex")
}

fn element<F: PrimeField>(value: &Value) -> F {
    F::from_canonical_u64(hex(value)).expect("vector values are canonical")
}

fn check_constants<F: PrimeField>(constants: &Value) {
    assert_eq!(hex(&constants["modulus"]), F::MODULUS);
    assert_eq!(
        constants["two_adicity"].as_u64(),
        Some(u64::from(F::TWO_ADICITY))
    );
    let roots = constants["root_of_unity_2_k"].as_array().expect("roots");
    assert_eq!(roots.len() as u32, F::TWO_ADICITY + 1);
    for (k, root) in roots.iter().enumerate() {
        assert_eq!(
            F::two_adic_generator(k as u32),
            element::<F>(root),
            "2^{k}-th root"
        );
    }
}

#[test]
fn constants_match_the_oracle() {
    let constants = vectors("constants.json");
    check_constants::<Goldilocks>(&constants["goldilocks"]);
    check_constants::<F65537>(&constants["f65537"]);
    assert_eq!(
        constants["goldilocks"]["multiplicative_generator"].as_u64(),
        Some(Goldilocks::MULTIPLICATIVE_GENERATOR.to_canonical_u64())
    );
    assert_eq!(
        constants["f65537"]["multiplicative_generator"].as_u64(),
        Some(F65537::MULTIPLICATIVE_GENERATOR.to_canonical_u64())
    );
    // ROOT_2_32 = 7^((P − 1) / 2^32), the value the GPU kernels and the verifier pin too.
    assert_eq!(
        Goldilocks::ROOT_2_32.to_canonical_u64(),
        0x1856_29dc_da58_878c
    );
}

#[test]
fn arithmetic_matches_the_oracle() {
    let file = vectors("goldilocks.json");
    let cases = file["cases"].as_array().expect("cases");
    assert!(
        cases.len() >= 10_000 + 49,
        "the oracle file has the edge pairs and 10,000 random cases"
    );
    for case in cases {
        let [a, b, sum, difference, product, inverse] = case.as_array().expect("case").as_slice()
        else {
            panic!("a case has six entries: {case}");
        };
        let (a, b): (Goldilocks, Goldilocks) = (element(a), element(b));
        assert_eq!(a + b, element(sum), "{a:?} + {b:?}");
        assert_eq!(a - b, element(difference), "{a:?} - {b:?}");
        assert_eq!(a * b, element(product), "{a:?} * {b:?}");
        match inverse {
            Value::Null => assert_eq!(a.inverse(), None, "zero has no inverse"),
            inverse => assert_eq!(a.inverse(), Some(element(inverse)), "{a:?}^-1"),
        }
    }
}
