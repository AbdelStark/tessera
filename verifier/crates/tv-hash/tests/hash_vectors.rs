// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tv-hash` against `spec/vectors/hash/hash.json` (RFC-0004 'Testing Strategy'), written by an
//! independent Python oracle with its own BLAKE3: every domain, every Merkle tree and canonical
//! opening, every column-digest matrix. And the envelope vectors of `spec/vectors/codec/`, whose
//! digests are `H("envelope", ·)`, opened with this crate's hash.

mod support;

use std::num::NonZeroU32;

use support::{Rng, digest_hex, list, number, text};
use tv_codec::{DecodeLimits, Envelope, ErrorCode, Expected};
use tv_field::Fp;
use tv_hash::column::column_digest;
use tv_hash::merkle::{node, verify_multiproof};
use tv_hash::{Digest, Domain, DomainHasher, hash};
use tv_testkit::{Json, blake3, hex, read_vectors};

/// The input of the domain vectors: byte `i` is `i mod 251`.
fn patterned(len: u64) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8).collect()
}

#[test]
fn every_domain_hashes_every_vector_input() {
    let file = read_vectors("hash/hash.json");
    let lengths: Vec<u64> = list(&file, "input_lengths")
        .iter()
        .map(|l| l.as_u64().unwrap())
        .collect();
    assert_eq!(lengths, [0, 1, 64, 1025]);
    let domains = list(&file, "domains");
    assert_eq!(domains.len(), 18, "every domain of RFC-0004");
    // The vectors list the domains in RFC-0004's table order, as `Domain::ALL` does.
    for (domain, vector) in Domain::ALL.into_iter().zip(domains) {
        assert_eq!(domain.name(), text(vector, "name"));
        assert_eq!(domain.context(), text(vector, "context"));
        let digests = list(vector, "digests");
        assert_eq!(digests.len(), lengths.len());
        for (&len, expected) in lengths.iter().zip(digests) {
            let input = patterned(len);
            let expected = digest_hex(expected.as_str().unwrap());
            assert_eq!(hash(domain, &input), expected, "{domain:?}, length {len}");
            // The same bytes in pieces: one byte at a time, and split at every block boundary.
            let mut bytewise = DomainHasher::new(domain);
            for byte in &input {
                bytewise.update(std::slice::from_ref(byte));
            }
            assert_eq!(bytewise.finalize(), expected, "{domain:?} bytewise, {len}");
            let mut blockwise = DomainHasher::new(domain);
            for block in input.chunks(64) {
                blockwise.update(block);
            }
            assert_eq!(
                blockwise.finalize(),
                expected,
                "{domain:?} blockwise, {len}"
            );
        }
    }
}

/// RFC-0004 'Testing Strategy': the context strings are distinct and `H(d1, x) ≠ H(d2, x)`.
#[test]
fn domains_are_distinct_and_separate_their_outputs() {
    let mut rng = Rng::new(0xd0_4a1);
    let inputs = [Vec::new(), rng.bytes(1), rng.bytes(64), rng.bytes(3000)];
    let mut seen = std::collections::BTreeMap::new();
    for domain in Domain::ALL {
        assert!(seen.insert(domain.context(), domain).is_none());
        assert!(domain.context().bytes().all(|b| (0x20..=0x7e).contains(&b)));
    }
    for input in &inputs {
        let digests: std::collections::BTreeSet<Digest> =
            Domain::ALL.iter().map(|&d| hash(d, input)).collect();
        assert_eq!(digests.len(), Domain::ALL.len(), "length {}", input.len());
    }
}

#[test]
fn every_vector_tree_has_its_root_and_every_canonical_opening_verifies() {
    let file = read_vectors("hash/hash.json");
    let trees = list(&file, "merkle");
    assert_eq!(trees.len(), 5);
    let mut openings = 0;
    for tree in trees {
        let leaves: Vec<Digest> = list(tree, "leaves")
            .iter()
            .map(|l| digest_hex(l.as_str().unwrap()))
            .collect();
        let n = leaves.len() as u64;
        let root = digest_hex(text(tree, "root"));
        // The root, recomputed level by level with this crate's node hash.
        let mut level = leaves.clone();
        while level.len() > 1 {
            level = level
                .chunks(2)
                .map(|pair| node(&pair[0], &pair[1]))
                .collect();
        }
        assert_eq!(level[0], root, "the root of the tree of {n} leaves");
        for opening in list(tree, "openings") {
            let opened: Vec<(u64, Digest)> = list(opening, "indices")
                .iter()
                .map(|i| {
                    let i = i.as_u64().unwrap();
                    (i, leaves[i as usize])
                })
                .collect();
            let siblings: Vec<Digest> = list(opening, "siblings")
                .iter()
                .map(|s| digest_hex(s.as_str().unwrap()))
                .collect();
            assert_eq!(
                verify_multiproof(&root, n, &opened, &siblings),
                Ok(()),
                "tree of {n}, indices {:?}",
                opened.iter().map(|o| o.0).collect::<Vec<_>>()
            );
            openings += 1;
        }
    }
    assert_eq!(openings, 1 + 3 + 7 + 4 + 5, "every opening of the vectors");
}

/// The symbols of column `j` of the vector matrix `rows`.
fn column_of(rows: &[Vec<Fp>], j: usize) -> Vec<Fp> {
    rows.iter().map(|row| row[j]).collect()
}

/// `d_j` computed here from RFC-0004 'Column digests' with the test kit's reference BLAKE3.
fn reference_column(commitment_id: &Digest, j: u64, block: usize, symbols: &[Fp]) -> Digest {
    let mut chain = blake3::derive_key(
        "tessera/v1/column-init",
        &[&commitment_id[..], &j.to_le_bytes()].concat(),
    );
    for step in symbols.chunks(block) {
        let mut input = chain.to_vec();
        for symbol in step {
            input.extend_from_slice(&symbol.to_canonical().to_le_bytes());
        }
        chain = blake3::derive_key("tessera/v1/column-step", &input);
    }
    chain
}

#[test]
fn every_vector_matrix_has_its_column_leaves() {
    let file = read_vectors("hash/hash.json");
    let matrices = list(&file, "columns");
    assert_eq!(matrices.len(), 7);
    for matrix in matrices {
        let commitment_id = digest_hex(text(matrix, "commitment_id"));
        let columns = number(matrix, "columns") as usize;
        let block = number(matrix, "block") as u32;
        let rows: Vec<Vec<Fp>> = list(matrix, "rows")
            .iter()
            .map(|row| {
                let row = row.as_array().unwrap();
                assert_eq!(row.len(), columns);
                row.iter()
                    .map(|x| Fp::from_canonical(x.as_hex_u64().unwrap()).unwrap())
                    .collect()
            })
            .collect();
        let leaves = list(matrix, "leaves");
        assert_eq!(leaves.len(), columns);
        let b = NonZeroU32::new(block).unwrap();
        for (j, expected) in leaves.iter().enumerate() {
            let symbols = column_of(&rows, j);
            let expected = digest_hex(expected.as_str().unwrap());
            assert_eq!(
                column_digest(&commitment_id, j as u64, b, &symbols),
                expected,
                "m = {}, B = {block}, column {j}",
                rows.len()
            );
            assert_eq!(
                reference_column(&commitment_id, j as u64, block as usize, &symbols),
                expected
            );
        }
    }
}

/// RFC-0004 'Testing Strategy': with `m` not a multiple of `B`, changing any symbol changes the
/// leaf; so do the column index and the commitment id. Steps longer than one BLAKE3 chunk (more
/// than 124 rows) take the crate's other path, checked against the reference.
#[test]
fn every_symbol_column_and_commitment_reaches_the_digest() {
    let mut rng = Rng::new(0xc01_0a5);
    for (m, block) in [
        (7usize, 3u32),
        (33, 32),
        (40, 1),
        (300, 128),
        (1000, 500),
        (5, 9),
    ] {
        let commitment_id = rng.digest();
        let symbols: Vec<Fp> = (0..m).map(|_| Fp::reduce(rng.next())).collect();
        let b = NonZeroU32::new(block).unwrap();
        let digest = column_digest(&commitment_id, 3, b, &symbols);
        assert_eq!(
            digest,
            reference_column(&commitment_id, 3, block as usize, &symbols)
        );
        assert_ne!(digest, column_digest(&commitment_id, 4, b, &symbols));
        let mut other_id = commitment_id;
        other_id[31] ^= 1;
        assert_ne!(digest, column_digest(&other_id, 3, b, &symbols));
        for i in 0..m {
            let mut changed = symbols.clone();
            changed[i] += Fp::ONE;
            assert_ne!(
                digest,
                column_digest(&commitment_id, 3, b, &changed),
                "m = {m}, symbol {i}"
            );
        }
        // One row fewer is another chain, and so is one more.
        assert_ne!(
            digest,
            column_digest(&commitment_id, 3, b, &symbols[..m - 1])
        );
    }
}

/// The limits an envelope case reads under: its named RFC-0003 default, or the file's.
fn limits_of(case: &Json, file: &Json) -> DecodeLimits {
    match case.get("limits").and_then(Json::as_str) {
        Some("message") => DecodeLimits::MESSAGE,
        Some("integer-model") => DecodeLimits::INTEGER_MODEL,
        Some(other) => panic!("unknown limits {other}"),
        None => {
            let limits = file.get("limits").unwrap();
            DecodeLimits {
                max_total_bytes: number(limits, "max_total_bytes"),
                max_vec_len: number(limits, "max_vec_len"),
                max_depth: number(limits, "max_depth") as u16,
            }
        }
    }
}

/// RFC-0003 'Envelope': the digest is `H("envelope", header ‖ payload)`. Opened with this
/// crate's hash, every envelope vector fails or opens exactly as with the reference BLAKE3, the
/// accepted ones open to their payload, and the digest cases fail as `ERR-CODEC-DIGEST`.
#[test]
fn envelope_vectors_open_with_this_crates_envelope_digest() {
    let file = read_vectors("codec/envelope.json");
    let cases = list(&file, "cases");
    assert_eq!(cases.len(), 52);
    let (mut opened, mut digest_failures) = (0, 0);
    for case in cases {
        let bytes = hex(text(case, "hex"));
        let expected = Expected {
            schema_id: number(case, "schema") as u16,
            version: number(case, "version") as u16,
        };
        let limits = limits_of(case, &file);
        let ours = Envelope::open(&bytes, expected, &limits, |signed| {
            hash(Domain::Envelope, signed)
        })
        .map(|envelope| envelope.payload().to_vec());
        let reference = Envelope::open(&bytes, expected, &limits, |signed| {
            blake3::derive_key("tessera/v1/envelope", signed)
        })
        .map(|envelope| envelope.payload().to_vec());
        assert_eq!(ours, reference, "{}", text(case, "why"));
        if let Some(payload) = case.get("payload") {
            assert_eq!(ours, Ok(hex(payload.as_str().unwrap())));
            opened += 1;
        }
        if case.get("code").and_then(Json::as_str) == Some("ERR-CODEC-DIGEST") {
            let error = ours.unwrap_err();
            assert_eq!(
                (error.code(), error.offset()),
                (ErrorCode::Digest, number(case, "offset"))
            );
            digest_failures += 1;
        }
    }
    assert_eq!((opened, digest_failures), (5, 4));
}
