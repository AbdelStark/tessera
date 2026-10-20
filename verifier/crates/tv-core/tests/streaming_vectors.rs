// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0011's row polynomial, column chains and merged root, from independent fixture data.
//! Use the verifier's natural-order NTTs and dense pairwise Merkle reduction; no prover code.

use std::num::NonZeroU32;

use tv_field::{Fp, coset_ntt, intt};
use tv_hash::{Digest, Domain, hash};
use tv_testkit::{Json, read_vectors};

fn member<'a>(value: &'a Json, name: &str) -> &'a Json {
    value.get(name).expect("member")
}

fn list(value: &Json) -> &[Json] {
    value.as_array().expect("array")
}

fn number(value: &Json) -> u64 {
    value.as_u64().expect("integer")
}

fn digest(value: &Json) -> Digest {
    let text = value.as_str().expect("digest");
    assert_eq!(text.len(), 64);
    std::array::from_fn(|i| u8::from_str_radix(&text[2 * i..2 * i + 2], 16).expect("byte"))
}

fn elements(value: &Json) -> Vec<Fp> {
    list(value)
        .iter()
        .map(|v| Fp::from_canonical(v.as_hex_u64().expect("hex")).expect("canonical"))
        .collect()
}

#[test]
fn all_streaming_fixture_codewords_chains_and_roots_agree() {
    let file = read_vectors("pcs-streaming/streaming.json");
    let cases = list(member(&file, "cases"));
    assert_eq!(cases.len(), 16);
    for case in cases {
        let name = member(case, "name").as_str().expect("name");
        let k = 1usize << number(member(member(case, "layout"), "log2_k"));
        let n = 2 * k;
        let rows = list(member(case, "rows"));
        let masks = list(member(case, "masks"));
        let expected = list(member(case, "codewords"));
        assert_eq!((rows.len(), masks.len()), (expected.len(), expected.len()));
        let mut codewords = Vec::new();
        for ((row, mask), expected) in rows.iter().zip(masks).zip(expected) {
            // g = f - m + X^k m, with f interpolated on H; evaluate on 7.L.
            let mut coefficients = elements(row);
            assert_eq!(coefficients.len(), k);
            intt(&mut coefficients).expect("inverse transform");
            coefficients.resize(n, Fp::ZERO);
            for (i, value) in elements(mask).into_iter().enumerate() {
                coefficients[i] -= value;
                coefficients[k + i] += value;
            }
            coset_ntt(&mut coefficients, Fp::from_canonical(7).expect("shift")).expect("coset");
            assert_eq!(coefficients, elements(expected), "{name}: codeword");
            codewords.push(coefficients);
        }
        let id = digest(member(case, "commitment_id"));
        let block = NonZeroU32::new(u32::try_from(number(member(case, "row_block"))).expect("u32"))
            .expect("B");
        let mut parts = Vec::new();
        let mut start = 0;
        for (indices, expected) in list(member(case, "segments"))
            .iter()
            .zip(list(member(case, "segment_digests")))
        {
            let end = start + list(indices).len();
            let columns: Vec<_> = (0..n)
                .map(|column| {
                    let symbols: Vec<_> = codewords[start..end].iter().map(|r| r[column]).collect();
                    tv_hash::column::column_digest(&id, column as u64, block, &symbols)
                })
                .collect();
            let expected: Vec<_> = list(expected).iter().map(digest).collect();
            assert_eq!(columns, expected, "{name}: segment chains");
            parts.push(columns);
            start = end;
        }
        assert_eq!(start, codewords.len());
        let mut leaves: Vec<_> = (0..n)
            .map(|column| {
                let input: Vec<_> = parts.iter().flat_map(|p| p[column]).collect();
                hash(Domain::ColumnMerge, &input)
            })
            .collect();
        let expected: Vec<_> = list(member(case, "leaves")).iter().map(digest).collect();
        assert_eq!(leaves, expected, "{name}: column merge");
        while leaves.len() > 1 {
            leaves = leaves
                .chunks_exact(2)
                .map(|p| tv_hash::merkle::node(&p[0], &p[1]))
                .collect();
        }
        assert_eq!(leaves, [digest(member(case, "root"))], "{name}: root");
    }
}
