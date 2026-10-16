// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `spec/vectors/poly/poly.json`, written by an independent oracle (`generate.py`, Python
//! big integers, every value from its definition): evaluations and bindings of `Fp` and `Fp2`
//! tables, `eq` tables, `LT(r, a)`, `LT2` and the causal selector, and round polynomials.

use std::fs;
use std::path::Path;

use serde_json::Value;
use tessera_field::{Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_poly::{
    RoundPoly, bind_lowest, bind_lowest_in_place, causal_selector, eq, eq_table, evaluate,
    evaluate_base, less_than, less_than_points,
};

type Fp = Goldilocks;
type Fp2 = GoldilocksExt2;

fn load() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/poly/poly.json");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).expect("the vector file is JSON")
}

fn hex(value: &Value) -> u64 {
    let text = value.as_str().expect("values are hex strings");
    u64::from_str_radix(text.trim_start_matches("0x"), 16).expect("valid hex")
}

fn fp(value: &Value) -> Fp {
    Fp::from_canonical_u64(hex(value)).expect("vector values are canonical")
}

fn fp2(value: &Value) -> Fp2 {
    let [c0, c1] = value
        .as_array()
        .expect("an Fp2 value is [c0, c1]")
        .as_slice()
    else {
        panic!("an Fp2 value has two coefficients: {value}");
    };
    Fp2::new(fp(c0), fp(c1))
}

fn list<T>(value: &Value, element: impl Fn(&Value) -> T) -> Vec<T> {
    value
        .as_array()
        .expect("a list")
        .iter()
        .map(element)
        .collect()
}

fn section<'a>(file: &'a Value, name: &str) -> &'a [Value] {
    let cases = file[name]
        .as_array()
        .unwrap_or_else(|| panic!("section {name}"));
    assert!(!cases.is_empty(), "section {name} has cases");
    cases
}

#[test]
fn the_file_is_for_goldilocks_and_its_extension() {
    let file = load();
    assert_eq!(hex(&file["modulus"]), Fp::MODULUS);
    assert_eq!(
        file["non_residue"].as_u64(),
        Some(<Fp2 as QuadraticExtension<Fp>>::NON_RESIDUE.to_canonical_u64())
    );
}

#[test]
fn table_evaluations_and_bindings_match_the_oracle() {
    let file = load();
    let (mut fp_sizes, mut fp2_sizes) = (Vec::new(), Vec::new());
    for case in section(&file, "multilinear") {
        let n = case["n"].as_u64().expect("n") as usize;
        let point = list(&case["point"], fp2);
        let value = fp2(&case["value"]);
        let bound = case.get("bound").map(|b| list(b, fp2));
        assert_eq!(point.len(), n);
        match case["field"].as_str() {
            Some("Fp") => {
                let table = list(&case["table"], fp);
                assert_eq!(table.len(), 1 << n);
                assert_eq!(evaluate_base(&table, &point), Ok(value), "Fp, 2^{n}");
                if let Some(bound) = bound {
                    assert_eq!(bind_lowest(&table, point[0]), Ok(bound), "Fp, 2^{n}");
                }
                fp_sizes.push(n);
            }
            Some("Fp2") => {
                let table = list(&case["table"], fp2);
                assert_eq!(table.len(), 1 << n);
                assert_eq!(evaluate(&table, &point), Ok(value), "Fp2, 2^{n}");
                if let Some(bound) = bound {
                    let mut in_place = table.clone();
                    bind_lowest_in_place(&mut in_place, point[0]).expect("a variable");
                    assert_eq!(in_place, bound, "Fp2, 2^{n}");
                }
                fp2_sizes.push(n);
            }
            other => panic!("unknown field {other:?}"),
        }
    }
    // The README promises these sizes.
    assert_eq!(fp_sizes, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]);
    assert_eq!(fp2_sizes, (0..=9).collect::<Vec<_>>());
}

#[test]
fn eq_tables_match_the_oracle() {
    let file = load();
    let cases = section(&file, "eq_tables");
    assert_eq!(cases.len(), 11, "points of 0 to 10 coordinates");
    for case in cases {
        let point = list(&case["point"], fp2);
        let table = list(&case["table"], fp2);
        assert_eq!(eq_table(&point), Ok(table), "{} coordinates", point.len());
    }
}

#[test]
fn less_than_matches_the_oracle() {
    let file = load();
    for case in section(&file, "less_than") {
        let r = list(&case["point"], fp2);
        let values = case["values"].as_array().expect("values");
        assert!(!values.is_empty());
        for pair in values {
            let a = hex(&pair[0]);
            assert_eq!(
                less_than(&r, a),
                Ok(fp2(&pair[1])),
                "LT(r, {a}), {} bits",
                r.len()
            );
        }
    }
}

#[test]
fn less_than_points_and_the_causal_selector_match_the_oracle() {
    let file = load();
    let cases = section(&file, "less_than_points");
    assert_eq!(cases.len(), 11, "0 to 10 bits per position");
    for case in cases {
        let (q, i) = (list(&case["q"], fp2), list(&case["i"], fp2));
        let n = q.len();
        assert_eq!(
            less_than_points(&i, &q),
            Ok(fp2(&case["lt_iq"])),
            "{n} bits"
        );
        assert_eq!(
            less_than_points(&q, &i),
            Ok(fp2(&case["lt_qi"])),
            "{n} bits"
        );
        assert_eq!(
            causal_selector(&q, &i),
            Ok(fp2(&case["causal"])),
            "{n} bits"
        );
        assert_eq!(eq(&q, &i), Ok(fp2(&case["eq"])), "{n} bits");
    }
}

#[test]
fn round_polynomials_match_the_oracle() {
    let file = load();
    let mut degrees = Vec::new();
    for case in section(&file, "round_polys") {
        let degree = case["degree"].as_u64().expect("degree") as usize;
        let sent = list(&case["evaluations"], fp2);
        let poly = RoundPoly::new(&sent).expect("a supported degree");
        assert_eq!(poly.degree(), degree);
        let value = poly.evaluate(fp2(&case["claim"]), fp2(&case["r"]));
        assert_eq!(value, fp2(&case["value"]), "degree {degree}");
        degrees.push(degree);
    }
    degrees.dedup();
    assert_eq!(degrees, (1..=7).collect::<Vec<_>>());
}
