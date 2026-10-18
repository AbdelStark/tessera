// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `spec/vectors/encoding/encoding.json`, written by an independent oracle (`generate.py`,
//! Python big integers, every value from its definition): transforms on the subgroup and on the
//! coset `7·L`, and rows encoded with vanishing masks of every size the README lists.

mod common;

use std::fs;
use std::path::Path;

use common::{Fp, Replay};
use serde_json::Value;
use tessera_field::PrimeField;
use tessera_pcs::{COSET_SHIFT, NttPlan, RowEncoder};

fn load() -> Value {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors/encoding/encoding.json");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).expect("the vector file is JSON")
}

fn hex(value: &Value) -> u64 {
    let text = value.as_str().expect("values are hex strings");
    u64::from_str_radix(text.trim_start_matches("0x"), 16).expect("valid hex")
}

fn fps(value: &Value) -> Vec<Fp> {
    value
        .as_array()
        .expect("a list")
        .iter()
        .map(|x| Fp::from_canonical_u64(hex(x)).expect("vector values are canonical"))
        .collect()
}

fn size(case: &Value, name: &str) -> usize {
    case[name].as_u64().unwrap_or_else(|| panic!("{name}")) as usize
}

fn section<'a>(file: &'a Value, name: &str) -> &'a [Value] {
    let cases = file[name]
        .as_array()
        .unwrap_or_else(|| panic!("section {name}"));
    assert!(!cases.is_empty(), "section {name} has cases");
    cases
}

#[test]
fn the_file_is_for_goldilocks_and_the_coset_seven_l() {
    let file = load();
    assert_eq!(hex(&file["modulus"]), Fp::MODULUS);
    assert_eq!(file["generator"].as_u64(), Some(7));
    assert_eq!(
        file["coset_shift"].as_u64(),
        Some(COSET_SHIFT.to_canonical_u64())
    );
}

#[test]
fn transforms_match_the_oracle() {
    let file = load();
    let mut sizes = Vec::new();
    for case in section(&file, "ntt") {
        let n = size(case, "n");
        let coefficients = fps(&case["coefficients"]);
        let values = fps(&case["values"]);
        let coset_values = fps(&case["coset_values"]);
        assert_eq!(coefficients.len(), n);
        let plan = NttPlan::new(n).expect("a supported size");

        let mut forward = coefficients.clone();
        plan.ntt(&mut forward).expect("matching length");
        assert_eq!(forward, values, "ntt, {n} points");
        plan.intt(&mut forward).expect("matching length");
        assert_eq!(forward, coefficients, "intt, {n} points");

        let mut coset = coefficients.clone();
        plan.coset_ntt(&mut coset, COSET_SHIFT)
            .expect("matching length");
        assert_eq!(coset, coset_values, "coset_ntt, {n} points");
        plan.coset_intt(&mut coset, COSET_SHIFT)
            .expect("matching length");
        assert_eq!(coset, coefficients, "coset_intt, {n} points");
        sizes.push(n);
    }
    // The README promises these sizes.
    assert_eq!(sizes, (1..=8).map(|e| 1 << e).collect::<Vec<usize>>());
}

#[test]
fn encodings_match_the_oracle() {
    let file = load();
    let mut shapes = Vec::new();
    for case in section(&file, "encoding") {
        let (k, p) = (size(case, "k"), size(case, "mask_coeffs"));
        let f = fps(&case["f"]);
        let row = fps(&case["row"]);
        let mask = fps(&case["mask"]);
        let expected = fps(&case["codeword"]);
        assert_eq!(
            (f.len(), row.len(), mask.len(), expected.len()),
            (k, k, p, 2 * k)
        );

        // The row is f's values on H, in natural order.
        let mut values = f.clone();
        NttPlan::new(k)
            .expect("a supported size")
            .ntt(&mut values)
            .expect("matching length");
        assert_eq!(values, row, "row of f, k = {k}");

        let encoder = RowEncoder::new(k, 1, p).expect("valid parameters");
        let mut source = Replay::new(mask);
        let mut codeword = vec![Fp::default(); 2 * k];
        encoder
            .encode_row(&row, &mut source, &mut codeword)
            .expect("matching lengths");
        assert_eq!(source.drawn, p, "P draws, k = {k}");
        assert_eq!(codeword, expected, "codeword, k = {k}, P = {p}");
        shapes.push((k, p));
    }
    // The README promises rows of 2^0 to 2^6 values with P in {0, 1, k/2, k}.
    let mut promised = Vec::new();
    for log_k in 0..=6 {
        let k = 1usize << log_k;
        let mut counts = vec![0, 1, k / 2, k];
        counts.sort_unstable();
        counts.dedup();
        promised.extend(counts.into_iter().map(|p| (k, p)));
    }
    assert_eq!(shapes, promised);
}
