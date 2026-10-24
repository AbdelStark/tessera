// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The reference sampler against `spec/vectors/scoring/`, written by an independent Python AES
//! and `hashlib` (RFC-0017 'Sampler PRF', RFC-0061 'The stream', RFC-0009 Lemma 10).
//!
//! The `aes` crate picks its backend per target: AES-NI on x86_64, fixsliced software on aarch64
//! unless `--cfg aes_armv8` selects the ARMv8 instructions. The same vectors pin every backend.

use tessera_codec::{HeapKey, Secret};
use tessera_scoring::{ClassBits, Sampler, SamplerBaseKey, SamplerError, SamplerKey, inverse_cdf};

fn key(hex: &str, bits: ClassBits) -> SamplerKey {
    assert_eq!(hex.len(), 32);
    let bytes: [u8; 16] =
        core::array::from_fn(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap());
    SamplerKey::new(Secret::new(HeapKey::from_box(Box::new(bytes))), bits)
}

/// A prompt column: comma-separated tokens, `-` for the empty prompt.
fn prompt(column: &str) -> Secret<Vec<u32>> {
    let tokens = match column {
        "-" => Vec::new(),
        list => list.split(',').map(|x| x.parse().unwrap()).collect(),
    };
    Secret::new(tokens)
}

fn rows(text: &str) -> impl Iterator<Item = Vec<&str>> {
    text.lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| line.split_whitespace().collect())
}

#[test]
fn every_draw_matches_the_independent_aes() {
    let mut count = 0;
    for row in rows(include_str!("../../../spec/vectors/scoring/prf.txt")) {
        let [key_hex, bits, tokens, t, _class, _ciphertext, u] = row[..] else {
            panic!("a PRF row has seven columns: {row:?}");
        };
        let bits = ClassBits::new(bits.parse().unwrap()).unwrap();
        let sampler = Sampler::new(&key(key_hex, bits));
        let class = sampler.class(&prompt(tokens));
        assert_eq!(
            sampler.draw(class, t.parse().unwrap()),
            u.parse::<u32>().unwrap(),
            "row {count}: key {key_hex}, b {}, position {t}",
            bits.get()
        );
        count += 1;
    }
    assert_eq!(count, 369);
}

/// A pin's PRF key from its base key, nonce, class bits and start (RFC-0061 'The sampler key and
/// its pins'), end to end through the public path: the derived key draws what the oracle's key
/// draws, and the class and draw of the prompt `1, 2, 3` at `t = 0` are the oracle's.
#[test]
fn every_derived_key_matches_the_independent_sha_256() {
    let mut count = 0;
    for row in rows(include_str!("../../../spec/vectors/scoring/derive.txt")) {
        let [base, nonce, bits, from_ns, derived, _class, u_0] = row[..] else {
            panic!("a derivation row has seven columns: {row:?}");
        };
        let base: [u8; 16] =
            core::array::from_fn(|i| u8::from_str_radix(&base[2 * i..2 * i + 2], 16).unwrap());
        let base = SamplerBaseKey::new(Secret::new(HeapKey::from_box(Box::new(base))));
        let nonce: [u8; 32] =
            core::array::from_fn(|i| u8::from_str_radix(&nonce[2 * i..2 * i + 2], 16).unwrap());
        let bits = ClassBits::new(bits.parse().unwrap()).unwrap();
        let ours = Sampler::new(&base.derive(&nonce, bits, from_ns.parse().unwrap()));
        let theirs = Sampler::new(&key(derived, bits));
        let prompt = Secret::new(vec![1, 2, 3]);
        let class = ours.class(&prompt);
        assert_eq!(
            ours.draw(class, 0),
            u_0.parse::<u32>().unwrap(),
            "row {count}"
        );
        let other = theirs.class(&prompt);
        for t in [0, 1, u32::MAX] {
            assert_eq!(ours.draw(class, t), theirs.draw(other, t), "row {count}");
        }
        count += 1;
    }
    assert_eq!(count, 46);
}

#[test]
fn every_token_matches_the_independent_bisection() {
    let mut count = 0;
    for row in rows(include_str!(
        "../../../spec/vectors/scoring/inverse_cdf.txt"
    )) {
        let u: u32 = row[0].parse().unwrap();
        let weights: Vec<u64> = row[2..].iter().map(|w| w.parse().unwrap()).collect();
        let expected = match row[1] {
            "none" => Err(SamplerError::NoMass),
            token => Ok(token.parse().unwrap()),
        };
        assert_eq!(
            inverse_cdf(&weights, u),
            expected,
            "u {u}, weights {weights:?}"
        );
        count += 1;
    }
    assert_eq!(count, 228);
}

/// SplitMix64 (Steele, Lea, Flood 2014), for seeded random distributions.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// RFC-0017's test plan: on 10^5 random distributions the token satisfies RFC-0009's bracket,
/// `C_< · 2^32 ≤ u·Z < C_≤ · 2^32`, which the `InverseCdf` certificate proves (Lemma 10), and has
/// a non-zero weight.
#[test]
fn the_token_satisfies_the_certificates_bracket() {
    let mut rng = SplitMix64(0x0017_0009);
    for _ in 0..100_000 {
        let size = 1 + (rng.next() % 64) as usize;
        let bits = 1 + rng.next() % 64;
        let zeros = rng.next() % 4;
        let weights: Vec<u64> = (0..size)
            .map(|_| {
                let w = rng.next() >> (64 - bits);
                if rng.next() % 4 < zeros { 0 } else { w }
            })
            .collect();
        let u = rng.next() as u32;
        let total: u128 = weights.iter().map(|&w| u128::from(w)).sum();
        let Ok(v) = inverse_cdf(&weights, u) else {
            assert_eq!(total, 0);
            continue;
        };
        let below: u128 = weights[..v].iter().map(|&w| u128::from(w)).sum();
        let through = below + u128::from(weights[v]);
        assert_ne!(weights[v], 0);
        assert!(below << 32 <= u128::from(u) * total);
        assert!(u128::from(u) * total < through << 32);
    }
}
