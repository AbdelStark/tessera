// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `spec/vectors/hash/hash.json`, written by an independent oracle (`generate.py`, with its own
//! BLAKE3 checked against the official vectors): every domain's context string and digests,
//! Merkle roots and canonical multiproofs, and chained column digests.

mod common;

use std::fs;
use std::num::NonZeroUsize;

use serde_json::Value;
use tessera_field::{Goldilocks, PrimeField};
use tessera_hash::{ColumnChains, Digest, Domain, MerkleTree, hash, verify_multiproof};

fn load() -> Value {
    let path = common::vectors().join("hash/hash.json");
    serde_json::from_str(&fs::read_to_string(path).expect("vector file")).expect("JSON")
}

fn digests(value: &Value) -> Vec<Digest> {
    value
        .as_array()
        .expect("a list of digests")
        .iter()
        .map(|d| common::digest(d.as_str().expect("hex")))
        .collect()
}

#[test]
fn every_domain_matches_the_oracle() {
    let file = load();
    let lengths: Vec<u64> = file["input_lengths"]
        .as_array()
        .expect("input lengths")
        .iter()
        .map(|n| n.as_u64().expect("length"))
        .collect();
    assert!(!lengths.is_empty());
    let domains = file["domains"].as_array().expect("domains");
    // The same domains, in the same order: RFC-0004's table.
    assert_eq!(domains.len(), Domain::ALL.len());
    for (entry, domain) in domains.iter().zip(Domain::ALL) {
        let name = entry["name"].as_str().expect("name");
        assert_eq!(domain.context(), format!("tessera/v1/{name}"));
        assert_eq!(entry["context"].as_str(), Some(domain.context()));
        let expected = digests(&entry["digests"]);
        assert_eq!(expected.len(), lengths.len(), "{name}");
        for (&len, expected) in lengths.iter().zip(expected) {
            let input: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            assert_eq!(hash(domain, &input), expected, "{name}, input length {len}");
        }
    }
}

#[test]
fn merkle_roots_and_multiproofs_match_the_oracle() {
    let file = load();
    let trees = file["merkle"].as_array().expect("trees");
    assert!(!trees.is_empty());
    for tree in trees {
        let leaves = digests(&tree["leaves"]);
        let n = leaves.len() as u64;
        let built = MerkleTree::new(leaves.clone()).expect("a power of two");
        let root = common::digest(tree["root"].as_str().expect("root"));
        assert_eq!(built.root(), root, "root of {n} leaves");
        let openings = tree["openings"].as_array().expect("openings");
        assert!(!openings.is_empty(), "{n} leaves");
        for opening in openings {
            let indices: Vec<u64> = opening["indices"]
                .as_array()
                .expect("indices")
                .iter()
                .map(|i| i.as_u64().expect("index"))
                .collect();
            let proof = built.open(&indices).expect("valid indices");
            assert_eq!(
                proof.siblings,
                digests(&opening["siblings"]),
                "{n} leaves, {indices:?}"
            );
            let opened: Vec<(u64, Digest)> =
                indices.iter().map(|&i| (i, leaves[i as usize])).collect();
            assert!(verify_multiproof(&root, n, &opened, &proof));
        }
    }
}

#[test]
fn column_chains_match_the_oracle() {
    let file = load();
    let matrices = file["columns"].as_array().expect("matrices");
    assert!(!matrices.is_empty());
    for matrix in matrices {
        let columns = matrix["columns"].as_u64().expect("columns") as usize;
        let block = matrix["block"].as_u64().expect("block") as usize;
        let rows: Vec<Goldilocks> = matrix["rows"]
            .as_array()
            .expect("rows")
            .iter()
            .flat_map(|row| row.as_array().expect("row").iter())
            .map(|x| {
                let text = x.as_str().expect("hex value");
                let value = u64::from_str_radix(text.trim_start_matches("0x"), 16).expect("hex");
                Goldilocks::from_canonical_u64(value).expect("canonical")
            })
            .collect();
        let commitment_id = common::digest(matrix["commitment_id"].as_str().expect("id"));
        let mut chains = ColumnChains::new(
            &commitment_id,
            NonZeroUsize::new(columns).expect("columns"),
            NonZeroUsize::new(block).expect("block"),
        );
        for chunk in rows.chunks(block * columns) {
            chains.absorb_block(chunk).expect("a valid block");
        }
        assert_eq!(
            chains.finish(),
            digests(&matrix["leaves"]),
            "{} rows, {columns} columns, block {block}",
            rows.len() / columns
        );
    }
}
