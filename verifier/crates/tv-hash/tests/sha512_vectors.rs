// SPDX-License-Identifier: Apache-2.0 OR MIT

//! SHA-512 against NIST's byte-oriented CAVP vectors in `spec/vectors/sha512/`: every short and
//! long message, and the 100 rounds of the Monte Carlo test as `spec/vectors/sha512/README.md`
//! describes them.

mod support;

use support::{Rng, rsp_fields};
use tv_hash::sha512::{Digest512, Sha512, sha512};
use tv_testkit::hex;

/// A 64-byte digest from hexadecimal.
fn digest_hex(text: &str) -> Digest512 {
    hex(text)
        .try_into()
        .unwrap_or_else(|_| panic!("{text} is not 64 bytes"))
}

/// The `(message, digest)` pairs of a message file; `Len` is in bits, and the empty message is
/// written `00`.
fn messages(rel: &str) -> Vec<(Vec<u8>, Digest512)> {
    let fields = rsp_fields(rel);
    let mut out = Vec::new();
    let mut rest = fields.as_slice();
    while let [(len, bits), (msg, data), (md, digest), tail @ ..] = rest {
        assert_eq!(
            (len.as_str(), msg.as_str(), md.as_str()),
            ("Len", "Msg", "MD")
        );
        let bits: usize = bits.parse().unwrap();
        assert_eq!(bits % 8, 0, "byte-oriented vectors");
        let mut message = hex(data);
        message.truncate(bits / 8);
        assert_eq!(message.len(), bits / 8);
        out.push((message, digest_hex(digest)));
        rest = tail;
    }
    assert!(rest.is_empty(), "{rel}: fields left over");
    out
}

/// One message through the one-shot function and through the hasher in several splittings.
fn check(message: &[u8], expected: &Digest512, rng: &mut Rng) {
    assert_eq!(sha512(message), *expected, "{} bytes", message.len());
    let mut pieces = Sha512::new();
    let mut rest = message;
    while !rest.is_empty() {
        let take = 1 + rng.index(rest.len().min(300));
        pieces.update(&rest[..take]);
        rest = &rest[take..];
    }
    assert_eq!(
        pieces.finalize(),
        *expected,
        "{} bytes in pieces",
        message.len()
    );
    if message.len() <= 256 {
        let mut bytewise = Sha512::new();
        for byte in message {
            bytewise.update(std::slice::from_ref(byte));
        }
        assert_eq!(
            bytewise.finalize(),
            *expected,
            "{} bytes bytewise",
            message.len()
        );
    }
}

#[test]
fn every_short_message() {
    let cases = messages("sha512/SHA512ShortMsg.rsp");
    assert_eq!(cases.len(), 129);
    let mut rng = Rng::new(0x5a512);
    for (index, (message, digest)) in cases.iter().enumerate() {
        assert_eq!(
            message.len(),
            index,
            "the short messages are 0 to 128 bytes"
        );
        check(message, digest, &mut rng);
    }
}

#[test]
fn every_long_message() {
    let cases = messages("sha512/SHA512LongMsg.rsp");
    assert_eq!(cases.len(), 128);
    let mut rng = Rng::new(0x5a513);
    for (message, digest) in &cases {
        check(message, digest, &mut rng);
    }
}

/// `MD0 = MD1 = MD2 = Seed`; `MDi = SHA-512(MD(i−3) ‖ MD(i−2) ‖ MD(i−1))` for `i` from 3 to 1002;
/// `MD1002` is the round's digest and the next round's seed.
#[test]
fn the_monte_carlo_test() {
    let fields = rsp_fields("sha512/SHA512Monte.rsp");
    let (name, seed) = &fields[0];
    assert_eq!(name, "Seed");
    let mut seed = digest_hex(seed);
    let rounds = &fields[1..];
    assert_eq!(rounds.len(), 200);
    for (round, pair) in rounds.chunks(2).enumerate() {
        assert_eq!(pair[0], ("COUNT".to_owned(), round.to_string()));
        assert_eq!(pair[1].0, "MD");
        let (mut a, mut b, mut c) = (seed, seed, seed);
        for _ in 3..=1002 {
            let next = sha512(&[a, b, c].concat());
            (a, b, c) = (b, c, next);
        }
        assert_eq!(c, digest_hex(&pair[1].1), "round {round}");
        seed = c;
    }
}

#[test]
fn padding_boundaries_agree_between_the_one_shot_function_and_the_hasher() {
    // Lengths where the padding needs a second block (112 to 127 bytes past a block boundary) or
    // fits the first exactly.
    let mut rng = Rng::new(0x5a514);
    let data = rng.bytes(700);
    for len in 0..data.len() {
        let mut hasher = Sha512::new();
        let (head, tail) = data[..len].split_at(len / 3);
        hasher.update(head).update(&[]).update(tail);
        assert_eq!(hasher.finalize(), sha512(&data[..len]), "length {len}");
        // `finalize` leaves the hasher usable.
        hasher.update(&data[len..len + 1]);
        assert_eq!(hasher.finalize(), sha512(&data[..len + 1]));
    }
}

/// FIPS 180-4's own examples (the NIST "Example Algorithms" document for SHA-512): "abc" and the
/// two-block message of 112 bytes, whose padding needs a second block.
#[test]
fn the_fips_examples() {
    assert_eq!(
        sha512(b"abc"),
        digest_hex(concat!(
            "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a",
            "2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
        ))
    );
    assert_eq!(
        sha512(
            b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopq\
              klmnopqrlmnopqrsmnopqrstnopqrstu"
        ),
        digest_hex(concat!(
            "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018",
            "501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909"
        ))
    );
}
