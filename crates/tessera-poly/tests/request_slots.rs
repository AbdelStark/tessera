// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Conformance and adversarial tests for RFC-0071 batched request-slot predicates
//! (`slot_selector`, `causal_selector_bounded`, `batch_causal_selector`) against
//! `spec/vectors/request-slots/slots.json`.

use std::path::PathBuf;

use serde_json::Value;
use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
use tessera_poly::{
    PolyError, batch_causal_selector, causal_selector, causal_selector_bounded, less_than,
    slot_selector,
};

fn fp2(value: &Value) -> GoldilocksExt2 {
    let arr = value.as_array().expect("Fp2 pair");
    GoldilocksExt2::new(
        Goldilocks::from_u64_reduced(arr[0].as_u64().expect("c0")),
        Goldilocks::from_u64_reduced(arr[1].as_u64().expect("c1")),
    )
}

fn point(value: &Value) -> Vec<GoldilocksExt2> {
    value.as_array().expect("point").iter().map(fp2).collect()
}

#[test]
fn request_slot_predicate_vectors_match_oracle() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../spec/vectors/request-slots/slots.json");
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(path).expect("slots.json"))
        .expect("valid json");
    let cases = doc["predicates"].as_array().expect("predicates array");
    assert!(!cases.is_empty());

    for case in cases {
        let name = case["name"].as_str().expect("name");
        let lengths: Vec<u64> = case["lengths"]
            .as_array()
            .expect("lengths")
            .iter()
            .map(|x| x.as_u64().expect("u64"))
            .collect();
        let heads = case["heads"].as_u64().expect("heads");
        let slot_bits = case["slot_bits"].as_u64().expect("slot_bits") as usize;
        let r_token = point(&case["r_token"]);
        let r_key = point(&case["r_key"]);
        let r_head = point(&case["r_head"]);

        let psi = slot_selector(&r_token, slot_bits, &lengths).expect(name);
        assert_eq!(psi, fp2(&case["slot_selector"]), "{name}: slot_selector");

        let head_lt = less_than(&r_head, heads).expect(name);
        let head_psi = head_lt * psi;
        assert_eq!(
            head_psi,
            fp2(&case["head_row_selector"]),
            "{name}: head_row_selector"
        );
        assert_eq!(
            GoldilocksExt2::ONE - head_psi,
            fp2(&case["softmax_root"]),
            "{name}: softmax_root"
        );

        let causal = batch_causal_selector(&r_token, &r_key, &lengths).expect(name);
        assert_eq!(
            causal,
            fp2(&case["causal_selector"]),
            "{name}: causal_selector"
        );
        assert_eq!(
            head_lt * causal,
            fp2(&case["head_causal_selector"]),
            "{name}: head_causal_selector"
        );
    }
}

#[test]
fn bounded_causal_and_slot_selectors_reject_malformed_shapes_and_bounds() {
    let f = |x: u64| GoldilocksExt2::from(Goldilocks::from_u64_reduced(x));
    let q = vec![f(3), f(7)];
    let i = vec![f(5), f(11)];

    // Full cube bound 2^2 = 4 matches standard causal_selector.
    assert_eq!(
        causal_selector_bounded(&q, &i, 4).unwrap(),
        causal_selector(&q, &i).unwrap()
    );
    assert_eq!(
        causal_selector_bounded(&q, &i, 0).unwrap(),
        GoldilocksExt2::ZERO
    );
    assert_eq!(
        causal_selector_bounded(&q, &i, 5),
        Err(PolyError::BoundTooLarge { bound: 5, vars: 2 })
    );
    assert_eq!(
        causal_selector_bounded(&q, &[f(1)], 2),
        Err(PolyError::PointLength {
            expected: 2,
            found: 1,
        })
    );

    // R = 3 requires slot_bits + 2 coordinates on r_token.
    assert_eq!(
        slot_selector(&[f(1), f(2), f(3)], 2, &[3, 1, 4]),
        Err(PolyError::PointLength {
            expected: 4,
            found: 3,
        })
    );
    assert_eq!(
        slot_selector(&[f(1), f(2), f(3), f(4)], 2, &[3, 5, 4]),
        Err(PolyError::BoundTooLarge { bound: 5, vars: 2 })
    );
    assert_eq!(
        batch_causal_selector(&[f(1), f(2)], &i, &[3, 2]),
        Err(PolyError::PointLength {
            expected: 3,
            found: 2,
        })
    );
    assert_eq!(
        batch_causal_selector(&[f(1), f(2), f(3)], &i, &[3, 5]),
        Err(PolyError::BoundTooLarge { bound: 5, vars: 2 })
    );
}
