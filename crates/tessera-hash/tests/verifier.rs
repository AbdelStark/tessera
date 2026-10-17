// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The prover's hashing against the verifier's, `tv-hash`, written separately from RFC-0004
//! (SP-6), on seeded random inputs: domain hashes, Merkle openings, column digests and whole
//! transcript sessions must agree bit for bit. The shared vectors pin chosen cases; this covers
//! the shapes between them, and checks the verifier's portable BLAKE3 against the `blake3`
//! crate's SIMD code on every length class (RFC-0019 'Dependencies'). Prover crates may use
//! verifier crates as dev-dependencies only (DEP-2).

mod common;

use std::num::{NonZeroU32, NonZeroUsize};

use common::SplitMix64;
use tessera_codec::{Ascii, Encode, encode};
use tessera_field::{Goldilocks, PrimeField, QuadraticExtension};
use tessera_hash::{
    ColumnChains, Digest, Domain, Hasher, MerkleTree, Mode, ParamSetId, SessionContext, Transcript,
    hash, merkle_node,
};
use tv_field::Fp;
use tv_hash::merkle::MultiproofError;
use tv_hash::transcript as tv;

const CHUNK: usize = 1024;

/// Every domain on both sides, paired by RFC-0004's table order and checked by context string.
fn domain_pairs() -> Vec<(Domain, tv_hash::Domain)> {
    let pairs: Vec<_> = Domain::ALL.into_iter().zip(tv_hash::Domain::ALL).collect();
    assert_eq!(pairs.len(), 19);
    for (prover, verifier) in &pairs {
        assert_eq!(prover.context(), verifier.context());
    }
    pairs
}

fn random_bytes(rng: &mut SplitMix64, len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len + 8);
    while out.len() < len {
        out.extend_from_slice(&rng.next_u64().to_le_bytes());
    }
    out.truncate(len);
    out
}

/// `input` cut at `cuts` random places, empty pieces included.
fn pieces<'a>(rng: &mut SplitMix64, input: &'a [u8], cuts: usize) -> Vec<&'a [u8]> {
    let mut at: Vec<usize> = (0..cuts)
        .map(|_| rng.below(input.len() as u64 + 1) as usize)
        .collect();
    at.sort_unstable();
    let mut out = Vec::with_capacity(cuts + 1);
    let mut start = 0;
    for cut in at {
        out.push(&input[start..cut]);
        start = cut;
    }
    out.push(&input[start..]);
    out
}

/// Both hashes of `input`, one shot and in random pieces on each side, agree.
fn check_hash(rng: &mut SplitMix64, (prover, verifier): (Domain, tv_hash::Domain), input: &[u8]) {
    let expected = hash(prover, input);
    let len = input.len();
    assert_eq!(
        tv_hash::hash(verifier, input),
        expected,
        "{prover:?}, {len} bytes"
    );
    let cuts = 1 + rng.below(6) as usize;
    let mut split = tv_hash::DomainHasher::new(verifier);
    for piece in pieces(rng, input, cuts) {
        split.update(piece);
    }
    assert_eq!(
        split.finalize(),
        expected,
        "{prover:?}, {len} bytes in pieces"
    );
    let mut split = Hasher::new(prover);
    for piece in pieces(rng, input, cuts) {
        split.update(piece);
    }
    assert_eq!(
        split.finalize(),
        expected,
        "prover, {prover:?}, {len} bytes in pieces"
    );
}

/// Every length up to two chunks and a block, lengths on each side of the block, chunk, 16-chunk
/// and 64-chunk boundaries up to 1,025 chunks, and random lengths up to 4 MiB, in every domain
/// in turn, each also hashed in random pieces.
#[test]
fn domain_hashes_agree_on_every_length_class() {
    let mut rng = SplitMix64::new(0xD1FF_B1A3);
    let domains = domain_pairs();
    let longest = 1025 * CHUNK + 65;
    let input = random_bytes(&mut rng, longest);

    let mut lengths: Vec<usize> = (0..=2 * CHUNK + 65).collect();
    for chunks in [
        3, 4, 5, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 127, 128, 129, 255, 256, 257,
        1023, 1024, 1025,
    ] {
        for offset in [-65, -64, -63, -1, 0, 1, 63, 64, 65] {
            lengths.push((chunks * CHUNK).checked_add_signed(offset).unwrap());
        }
    }
    for (i, len) in lengths.into_iter().enumerate() {
        check_hash(&mut rng, domains[i % domains.len()], &input[..len]);
    }
    for i in 0..12 {
        let len = rng.below(4 << 20) as usize;
        let long = random_bytes(&mut rng, len);
        check_hash(&mut rng, domains[i % domains.len()], &long);
    }
    // Every domain on the shapes most hashes take: a Merkle node, a column step, a long message.
    for pair in domains {
        for len in [0, 1, 40, 64, 65, 1024, 1025, 16 * CHUNK + 1] {
            check_hash(&mut rng, pair, &input[..len]);
        }
    }
}

fn digests(rng: &mut SplitMix64, count: usize) -> Vec<Digest> {
    (0..count).map(|_| rng.digest()).collect()
}

/// Trees of 1 to 4,096 leaves: the verifier accepts every multiproof the prover opens, for
/// random sets of queries, a single leaf at either end and every leaf at once, and rejects it
/// once any sibling changes. This is the agreement on the canonical sibling order.
#[test]
fn verifier_accepts_every_prover_multiproof() {
    let mut rng = SplitMix64::new(0x3E4C_1E00);
    for log_n in 0..=12u32 {
        let n = 1u64 << log_n;
        let leaves = digests(&mut rng, n as usize);
        let tree = MerkleTree::new(leaves.clone()).unwrap();
        let root = tree.root();
        let mut queries: Vec<Vec<u64>> = vec![vec![0], vec![n - 1], (0..n).collect()];
        for _ in 0..24 {
            let count = 1 + rng.below(n.min(40)) as usize;
            queries.push((0..count).map(|_| rng.below(n)).collect());
        }
        for query in queries {
            let mut indices = query.clone();
            indices.sort_unstable();
            indices.dedup();
            let opened: Vec<(u64, Digest)> =
                indices.iter().map(|&i| (i, leaves[i as usize])).collect();
            let mut siblings = tree.open(&query).unwrap().siblings;
            assert_eq!(
                tv_hash::merkle::verify_multiproof(&root, n, &opened, &siblings),
                Ok(()),
                "n = {n}, indices {indices:?}"
            );
            if !siblings.is_empty() {
                let at = rng.below(siblings.len() as u64) as usize;
                siblings[at][rng.below(32) as usize] ^= 1 << rng.below(8);
                assert_eq!(
                    tv_hash::merkle::verify_multiproof(&root, n, &opened, &siblings),
                    Err(MultiproofError::RootMismatch),
                    "n = {n}, sibling {at} changed"
                );
            }
        }
    }
    for _ in 0..64 {
        let (left, right) = (rng.digest(), rng.digest());
        assert_eq!(
            tv_hash::merkle::node(&left, &right),
            merkle_node(&left, &right)
        );
    }
}

/// Matrices of 0 to 1,000 rows and 1 to 4 columns, in blocks of 1 row to more than a
/// chunk's worth: each column's digest on the verifier is the prover's leaf.
#[test]
fn column_digests_agree() {
    let mut rng = SplitMix64::new(0xC0_1D16);
    let shapes = [
        (0, 1, 1),
        (0, 3, 64),
        (1, 1, 1),
        (5, 2, 8),
        (8, 3, 8),
        (9, 3, 8),
        (123, 2, 124),
        (124, 2, 124),
        (125, 2, 124),
        (250, 1, 125),
        (300, 2, 128),
        (257, 4, 129),
        (1000, 2, 300),
        (999, 3, 1000),
        (64, 1, 1),
    ];
    for (rows, columns, block) in shapes {
        let commitment_id = rng.digest();
        let symbol = |rng: &mut SplitMix64| match rng.below(16) {
            0 => 0,
            1 => Goldilocks::MODULUS - 1,
            _ => rng.below(Goldilocks::MODULUS),
        };
        let matrix: Vec<u64> = (0..rows * columns).map(|_| symbol(&mut rng)).collect();
        let field: Vec<Goldilocks> = matrix
            .iter()
            .map(|&x| Goldilocks::from_canonical_u64(x).unwrap())
            .collect();
        let mut chains = ColumnChains::new(
            &commitment_id,
            NonZeroUsize::new(columns).unwrap(),
            NonZeroUsize::new(block).unwrap(),
        );
        for block_rows in field.chunks(block * columns) {
            chains.absorb_block(block_rows).unwrap();
        }
        for (j, leaf) in chains.finish().into_iter().enumerate() {
            let column: Vec<Fp> = matrix
                .iter()
                .skip(j)
                .step_by(columns)
                .map(|&x| Fp::from_canonical(x).unwrap())
                .collect();
            let b = NonZeroU32::new(block as u32).unwrap();
            assert_eq!(
                tv_hash::column::column_digest(&commitment_id, j as u64, b, &column),
                leaf,
                "{rows} rows, column {j} of {columns}, B = {block}"
            );
        }
    }
}

/// Printable ASCII of up to 24 bytes, mostly from the label alphabet of RFC-0014.
fn label(rng: &mut SplitMix64) -> String {
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789/-";
    let len = rng.below(25) as usize;
    (0..len)
        .map(|_| match rng.below(8) {
            0 => char::from(0x20 + rng.below(95) as u8),
            _ => char::from(ALPHABET[rng.below(ALPHABET.len() as u64) as usize]),
        })
        .collect()
}

/// Absorbs `value` on both sides: the prover streams its encoding, the verifier takes its TCE
/// bytes, as `tv-core` will from a decoded proof.
fn absorb_both<T: Encode>(
    prover: &mut Transcript,
    verifier: &mut tv::Transcript,
    label: &str,
    value: &T,
) {
    prover.absorb(label, value);
    verifier.absorb(tv::Ascii::new(label).unwrap(), &encode(value));
}

/// A random prover message, from a `u64` to a few hundred KiB, absorbed on both sides.
fn absorb_message(
    rng: &mut SplitMix64,
    prover: &mut Transcript,
    verifier: &mut tv::Transcript,
    label: &str,
) {
    match rng.below(5) {
        0 => absorb_both(prover, verifier, label, &rng.next_u64()),
        1 => absorb_both(prover, verifier, label, &rng.digest()),
        2 => {
            let len = rng.below(40) as usize;
            let value: Vec<Goldilocks> = (0..len)
                .map(|_| Goldilocks::from_canonical_u64(rng.below(Goldilocks::MODULUS)).unwrap())
                .collect();
            absorb_both(prover, verifier, label, &value);
        }
        3 => {
            let len = rng.below(3 * CHUNK as u64) as usize;
            absorb_both(prover, verifier, label, &random_bytes(rng, len));
        }
        _ => {
            // Long enough for the prover's staged absorb to pass several 64 KiB updates on.
            let len = 64 * CHUNK + rng.below(200 * CHUNK as u64) as usize;
            absorb_both(prover, verifier, label, &random_bytes(rng, len));
        }
    }
}

/// Whole sessions of random length and order: the same context, absorbs, squeezes, challenges,
/// index draws and proofs of work give the same state and counter after every step, and the same
/// values; a session may end on a proof-of-work check that both sides judge alike.
#[test]
fn transcript_sessions_agree() {
    let mut rng = SplitMix64::new(0x7A5C_0FF1);
    for session in 0..160 {
        let protocol = label(&mut rng);
        let params = label(&mut rng);
        let mode = rng.below(2) == 0;
        let statement_digest = rng.digest();
        let mut session_id = [0u8; 16];
        session_id.copy_from_slice(&rng.digest()[..16]);

        let context = SessionContext {
            protocol: Ascii::new(protocol.clone()).unwrap(),
            mode: if mode {
                Mode::Live
            } else {
                Mode::NonInteractive
            },
            params: ParamSetId(Ascii::new(params.clone()).unwrap()),
            statement_digest,
            session_id,
        };
        let verifier_context = tv::SessionContext {
            protocol: tv::Ascii::new(&protocol).unwrap(),
            mode: if mode {
                tv::Mode::Live
            } else {
                tv::Mode::NonInteractive
            },
            params: tv::ParamSetId(tv::Ascii::new(&params).unwrap()),
            statement_digest,
            session_id,
        };
        assert_eq!(
            verifier_context.to_tce(),
            encode(&context),
            "session {session}"
        );

        let mut prover = Transcript::new(&context);
        let mut verifier = tv::Transcript::new(&verifier_context);
        let steps = 1 + rng.below(30);
        for step in 0..steps {
            let at = format!("session {session}, step {step}");
            let name = label(&mut rng);
            let text = tv::Ascii::new(&name).unwrap();
            match rng.below(7) {
                0 | 1 => absorb_message(&mut rng, &mut prover, &mut verifier, &name),
                2 => assert_eq!(prover.squeeze(&name), verifier.squeeze(text), "{at}"),
                3 => assert_eq!(
                    prover.challenge_fp(&name).to_canonical_u64(),
                    verifier.challenge_fp(text).unwrap().to_canonical(),
                    "{at}"
                ),
                4 => {
                    let [c0, c1] = prover.challenge_fp2(&name).coeffs();
                    let value = verifier.challenge_fp2(text).unwrap();
                    assert_eq!(
                        [value.c0.to_canonical(), value.c1.to_canonical()],
                        [c0.to_canonical_u64(), c1.to_canonical_u64()],
                        "{at}"
                    );
                }
                5 => {
                    let log_n = rng.below(65) as u32;
                    let count = rng.below(41) as u32;
                    assert_eq!(
                        prover.challenge_indices(&name, log_n, count),
                        verifier.challenge_indices(text, log_n, count).unwrap(),
                        "{at}, log_n {log_n}, count {count}"
                    );
                }
                _ => {
                    let bits = rng.below(11) as u8;
                    let nonce = prover.grind(&name, bits);
                    verifier = verifier.check_grind(text, bits, nonce).unwrap();
                }
            }
            assert_eq!(verifier.state(), &prover.state(), "{at}");
            assert_eq!(verifier.counter(), prover.counter(), "{at}");
        }
        // A nonce nobody ground for: valid with probability 2^-12.
        let name = label(&mut rng);
        let nonce = rng.next_u64();
        let valid = prover.check_grind(&name, 12, nonce);
        match verifier.check_grind(tv::Ascii::new(&name).unwrap(), 12, nonce) {
            Ok(verifier) => {
                assert!(
                    valid,
                    "session {session}: only the verifier accepts {nonce}"
                );
                assert_eq!(verifier.state(), &prover.state());
            }
            Err(failed) => {
                assert!(!valid, "session {session}: only the prover accepts {nonce}");
                assert_eq!(failed.required(), 12);
            }
        }
    }
}
