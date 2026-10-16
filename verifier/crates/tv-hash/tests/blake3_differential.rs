// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tv-hash`'s BLAKE3 against the test kit's reference BLAKE3, which follows the specification's
//! structure: one compression per call, a chunk state and a stack merged chunk by chunk. The two
//! share no code.
//!
//! Inputs are seeded random bytes whose lengths sit on, before and after every block boundary of
//! the first chunk, every chunk boundary of the first 40 chunks, every power-of-two tree boundary
//! up to 2^11 chunks (2 MiB), and the boundaries of this crate's own structure (16 chunks hashed
//! side by side, subtrees of 64 chunks), with random lengths up to 1.25 MiB. Each input is hashed
//! in one call and in pieces, split at random points, at unaligned sizes and at chunk-aligned
//! sizes, so that subtrees start at every alignment.

mod support;

use support::Rng;
use tv_hash::{Domain, DomainHasher, hash};
use tv_testkit::blake3;

const KIB: usize = 1024;

fn lengths(rng: &mut Rng) -> Vec<usize> {
    let mut out = Vec::new();
    let mut around = |at: usize, offsets: &[isize]| {
        for &offset in offsets {
            if let Some(len) = at.checked_add_signed(offset) {
                out.push(len);
            }
        }
    };
    for block in 0..=16 {
        around(64 * block, &[-1, 0, 1]);
    }
    for chunk in 1..=40 {
        around(KIB * chunk, &[-1, 0, 1]);
    }
    for log in 1..=11 {
        around(KIB << log, &[-1025, -1024, -1, 0, 1, 64, 1023, 1024, 1025]);
    }
    // Groups of 16 chunks and subtrees of 64, then a few beyond 1 MiB.
    for groups in 1..=8 {
        around(16 * KIB * groups, &[-1, 0, 1]);
    }
    for subtrees in 1..=17 {
        around(64 * KIB * subtrees, &[-1, 1]);
    }
    for len in [
        KIB * KIB + 17 * KIB + 5,
        3 * KIB * KIB / 2,
        2 * KIB * KIB + 1,
    ] {
        out.push(len);
    }
    for _ in 0..40 {
        out.push(rng.index(1280 * KIB));
    }
    out.sort_unstable();
    out.dedup();
    out
}

#[test]
fn one_call_and_pieces_agree_with_the_reference() {
    let mut rng = Rng::new(0xb1a6e3);
    let lengths = lengths(&mut rng);
    let data = rng.bytes(*lengths.last().unwrap() + 4096);
    assert!(*lengths.last().unwrap() >= KIB * KIB, "past 1 MiB");
    for (case, &len) in lengths.iter().enumerate() {
        let domain = Domain::ALL[case % Domain::ALL.len()];
        // A different window of the data for each case.
        let start = rng.index(4096);
        let input = &data[start..start + len];
        let expected = blake3::derive_key(domain.context(), input);
        assert_eq!(hash(domain, input), expected, "{domain:?}, length {len}");

        // Random pieces, from a byte to tens of kilobytes.
        let mut hasher = DomainHasher::new(domain);
        let mut rest = input;
        while !rest.is_empty() {
            let take = match rng.below(4) {
                0 => 1 + rng.index(64),
                1 => 1 + rng.index(2 * KIB),
                _ => 1 + rng.index(80 * KIB),
            }
            .min(rest.len());
            hasher.update(&rest[..take]);
            rest = &rest[take..];
        }
        assert_eq!(
            hasher.finalize(),
            expected,
            "{domain:?}, {len} in random pieces"
        );

        // An unaligned head, then pieces of a fixed number of whole chunks: every subtree size
        // and alignment the hasher can choose.
        if case % 3 == 0 {
            let head = 1 + rng.index(3 * KIB);
            let piece = KIB * (1 + rng.index(70));
            let mut hasher = DomainHasher::new(domain);
            let (first, mut rest) = input.split_at(head.min(len));
            hasher.update(first);
            while !rest.is_empty() {
                let take = piece.min(rest.len());
                hasher.update(&rest[..take]);
                rest = &rest[take..];
            }
            assert_eq!(
                hasher.finalize(),
                expected,
                "{domain:?}, {len} in pieces of {piece} after {head}"
            );
        }
    }
}

/// `finalize` does not end the hasher: hashing a prefix, finalizing, then continuing gives the
/// hash of the whole, at every boundary the hasher keeps state across.
#[test]
fn finalize_leaves_the_hasher_usable() {
    let mut rng = Rng::new(0xf1a1);
    let data = rng.bytes(300 * KIB);
    for cut in [
        0,
        1,
        64,
        1023,
        1024,
        1025,
        16 * KIB,
        64 * KIB,
        64 * KIB + 1,
        200 * KIB,
    ] {
        let mut hasher = DomainHasher::new(Domain::Table);
        hasher.update(&data[..cut]);
        assert_eq!(
            hasher.finalize(),
            blake3::derive_key(Domain::Table.context(), &data[..cut])
        );
        hasher.update(&data[cut..]);
        assert_eq!(
            hasher.finalize(),
            blake3::derive_key(Domain::Table.context(), &data),
            "continued after {cut}"
        );
    }
}
