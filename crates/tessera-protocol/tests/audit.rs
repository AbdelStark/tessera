// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Audit sampling (RFC-0016), through the public API:
//! - each bucket's selected fraction over 10^6 synthetic requests lies in a 99.9% binomial
//!   interval of its rate, and candidates past every bucket are all selected;
//! - one bit of either share or of the manifest's digest changes about half the decisions;
//! - the boundaries: the threshold compared strictly, the first bucket that holds a length, a
//!   coin toss that does not open;
//! - batching: every request once, within the bounds RFC-0016 'Batching' promises, the same
//!   batches whatever the input's order, and first fit rather than best fit;
//! - `spec/vectors/audit/audit.json`, from an independent oracle: thresholds, seeds, selection
//!   values, samples with their encodings and envelopes, refused inputs at their offsets, and
//!   batches.

mod common;

use std::collections::BTreeMap;
use std::fs;

use proptest::prelude::*;
use serde_json::Value;
use tessera_codec::error::ErrorCode;
use tessera_codec::{Ascii, DecodeLimits, decode, encode, envelope, schema};
use tessera_hash::ParamSetId;
use tessera_protocol::{
    AuditPolicy, AuditSample, Batch, CoinTossTranscript, ProtocolError, RateBucket,
    RepeatedRequest, RequestKey, ShareRecord, batches, coin_commitment, sample_seed,
    selection_value,
};

/// SplitMix64, the inputs' generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn bytes<const N: usize>(&mut self) -> [u8; N] {
        let mut out = [0u8; N];
        for chunk in out.chunks_mut(8) {
            let word = self.next().to_le_bytes();
            chunk.copy_from_slice(&word[..chunk.len()]);
        }
        out
    }

    fn key(&mut self) -> RequestKey {
        RequestKey {
            session_id: self.bytes(),
            request_id: self.next(),
        }
    }
}

fn policy(buckets: Vec<RateBucket>, min_batch_tokens: u32, max_batch_tokens: u32) -> AuditPolicy {
    AuditPolicy::new(
        86_400_000_000_000,
        buckets,
        min_batch_tokens,
        max_batch_tokens,
        604_800_000_000_000,
        ParamSetId(Ascii::new("live-64/1").expect("ASCII")),
    )
    .expect("a valid policy")
}

fn bucket(max_tokens: u32, rate: &str) -> RateBucket {
    RateBucket {
        max_tokens,
        threshold: RateBucket::threshold_of(rate).expect("a rate"),
    }
}

fn toss(rng: &mut Rng) -> CoinTossTranscript {
    let verifier_share = rng.bytes();
    CoinTossTranscript {
        verifier_commitment: coin_commitment(&verifier_share),
        prover_share: rng.bytes(),
        verifier_share,
    }
}

#[test]
fn each_bucket_selects_at_its_rate() {
    let mut rng = Rng(0x0016_0001);
    let buckets = vec![
        bucket(1024, "0.1"),
        bucket(32_768, "0.02"),
        bucket(131_072, "0.004"),
    ];
    let policy = policy(buckets.clone(), 16_384, 262_144);
    let coin_toss = toss(&mut rng);
    let manifest = rng.bytes();
    // A quarter of the requests in each bucket, and a quarter past them all.
    let lengths = [512u64, 20_000, 100_000, 131_073];
    let mut candidates = Vec::with_capacity(1_000_000);
    for i in 0..1_000_000u64 {
        candidates.push((rng.key(), lengths[(i % 4) as usize]));
    }
    let sample = AuditSample::draw(manifest, policy, coin_toss, candidates.iter().copied())
        .expect("a coin toss that opens");
    let chosen: std::collections::BTreeSet<_> = sample.selected().iter().copied().collect();
    let mut drawn = [0u64; 4];
    for (key, tokens) in &candidates {
        if chosen.contains(key) {
            drawn[lengths.iter().position(|l| l == tokens).expect("a length")] += 1;
        }
    }
    let n = 250_000.0;
    // The two-sided 99.9% normal quantile.
    let z = 3.290_5;
    for (b, k) in buckets.iter().zip(drawn) {
        let p = b.threshold as f64 / 2f64.powi(64);
        let (mean, sd) = (n * p, (n * p * (1.0 - p)).sqrt());
        assert!(
            (k as f64 - mean).abs() <= z * sd,
            "bucket up to {}: {k} selected, expected {mean:.0} ± {:.0}",
            b.max_tokens,
            z * sd
        );
    }
    assert_eq!(
        drawn[3], 250_000,
        "every candidate past the buckets is selected"
    );
}

/// The fraction of `keys` whose decision differs under the seeds of two coin tosses over two
/// manifests, at rate 1/2.
fn changed(
    policy: &AuditPolicy,
    a: (&CoinTossTranscript, &[u8; 32]),
    b: (&CoinTossTranscript, &[u8; 32]),
    keys: &[RequestKey],
) -> f64 {
    let (sa, sb) = (sample_seed(a.0, a.1), sample_seed(b.0, b.1));
    let differ = keys
        .iter()
        .filter(|k| policy.selects(&sa, k, 1) != policy.selects(&sb, k, 1))
        .count();
    differ as f64 / keys.len() as f64
}

#[test]
fn one_bit_of_either_share_changes_half_the_decisions() {
    let mut rng = Rng(0x0016_0002);
    let half = policy(vec![bucket(u32::MAX, "0.5")], 1, 2);
    let keys: Vec<RequestKey> = (0..4096).map(|_| rng.key()).collect();
    let manifest: [u8; 32] = rng.bytes();
    let base = toss(&mut rng);
    // 4096 decisions: a fraction changed has a standard deviation of 1/128; ±0.06 is 7.7 of them.
    let close = |f: f64, what: &str| assert!((f - 0.5).abs() < 0.06, "{what}: {f:.3} changed");
    for byte in 0..32 {
        let bit = 1u8 << (byte % 8);
        let mut t = base.clone();
        t.prover_share[byte] ^= bit;
        close(
            changed(&half, (&base, &manifest), (&t, &manifest), &keys),
            &format!("s_P byte {byte}"),
        );
        let mut t = base.clone();
        t.verifier_share[byte] ^= bit;
        close(
            changed(&half, (&base, &manifest), (&t, &manifest), &keys),
            &format!("s_V byte {byte}"),
        );
        let mut m = manifest;
        m[byte] ^= bit;
        close(
            changed(&half, (&base, &manifest), (&base, &m), &keys),
            &format!("manifest byte {byte}"),
        );
    }
    // The shares are not interchangeable: swapping them is another seed.
    let mut swapped = base.clone();
    swapped.prover_share = base.verifier_share;
    swapped.verifier_share = base.prover_share;
    close(
        changed(&half, (&base, &manifest), (&swapped, &manifest), &keys),
        "the shares swapped",
    );
}

#[test]
fn the_threshold_is_compared_strictly() {
    let mut rng = Rng(0x0016_0003);
    let seed = rng.bytes();
    let key = rng.key();
    let value = selection_value(&seed, &key);
    let at = |threshold| {
        policy(
            vec![RateBucket {
                max_tokens: 10,
                threshold,
            }],
            1,
            2,
        )
    };
    assert!(
        !at(value).selects(&seed, &key, 10),
        "a value equal to the threshold"
    );
    assert!(
        at(value + 1).selects(&seed, &key, 10),
        "a value just below it"
    );
    assert!(
        !at(0).selects(&seed, &key, 0),
        "a threshold of 0 selects nothing"
    );
}

#[test]
fn a_length_takes_the_first_bucket_that_holds_it() {
    let p = policy(
        vec![
            RateBucket {
                max_tokens: 100,
                threshold: 1,
            },
            RateBucket {
                max_tokens: 200,
                threshold: 2,
            },
        ],
        1,
        2,
    );
    assert_eq!(p.threshold(0), Some(1));
    assert_eq!(p.threshold(100), Some(1));
    assert_eq!(p.threshold(101), Some(2));
    assert_eq!(p.threshold(200), Some(2));
    assert_eq!(p.threshold(201), None);
    assert_eq!(p.threshold(u64::MAX), None);
    // Past every bucket, selected whatever the seed, even when every rate is 0.
    let zero = policy(
        vec![RateBucket {
            max_tokens: 100,
            threshold: 0,
        }],
        1,
        2,
    );
    let mut rng = Rng(0x0016_0004);
    for _ in 0..64 {
        let (seed, key) = (rng.bytes(), rng.key());
        assert!(zero.selects(&seed, &key, 101));
        assert!(!zero.selects(&seed, &key, 100));
    }
    // No buckets: every candidate is past them all.
    let none = policy(Vec::new(), 1, 2);
    assert!(none.selects(&[0; 32], &rng.key(), 0));
}

#[test]
fn a_reveal_that_does_not_open_its_commitment_is_refused() {
    let mut rng = Rng(0x0016_0005);
    let mut coin_toss = toss(&mut rng);
    assert_eq!(coin_toss.check(), Ok(()));
    coin_toss.verifier_share[31] ^= 1;
    let p = policy(vec![bucket(10, "1")], 1, 2);
    let refused = AuditSample::draw(rng.bytes(), p, coin_toss.clone(), [(rng.key(), 1)]);
    assert_eq!(refused, Err(ProtocolError::CoinToss));
    assert_eq!(ProtocolError::CoinToss.code(), ErrorCode::ProtoCointoss);
    // The commitment is H("coin-commit", s_V), not of s_P.
    let mut wrong = toss(&mut rng);
    wrong.verifier_commitment = coin_commitment(&wrong.prover_share);
    assert_eq!(wrong.check(), Err(ProtocolError::CoinToss));
}

#[test]
fn the_sample_is_sorted_and_counts_each_key_once() {
    let mut rng = Rng(0x0016_0006);
    let all = policy(vec![bucket(u32::MAX, "1")], 1, 2);
    let keys: Vec<RequestKey> = (0..50).map(|_| rng.key()).collect();
    let mut given: Vec<(RequestKey, u64)> = keys.iter().rev().map(|k| (*k, 3)).collect();
    given.extend(keys.iter().map(|k| (*k, 4)));
    let sample = AuditSample::draw(rng.bytes(), all, toss(&mut rng), given).expect("a sample");
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(sample.selected(), &sorted[..]);
    // Its envelope opens to it.
    let sealed = envelope::seal(&schema::AUDIT_SAMPLE, &sample);
    let opened: AuditSample =
        envelope::open(&sealed, &schema::AUDIT_SAMPLE, DecodeLimits::MESSAGE).expect("opens");
    assert_eq!(opened, sample);
}

#[test]
fn rates_read_exactly() {
    for (rate, threshold) in [
        ("0", 0),
        ("1", u64::MAX),
        ("1.0000000000000000000", u64::MAX),
        ("0.5", 1 << 63),
        ("0.25", 1 << 62),
        ("0.1", 1_844_674_407_370_955_161),
        ("0.0000000000000000001", 1),
        ("0.0000000000000000000", 0),
    ] {
        assert_eq!(RateBucket::threshold_of(rate), Ok(threshold), "{rate}");
    }
    for rate in [
        "",
        ".5",
        "0.",
        "2",
        "1.1",
        "0.12345678901234567890",
        "-0",
        "0,5",
        "1e-1",
        " 0",
        "00.1",
        "0.1 ",
        "+0.1",
    ] {
        assert!(RateBucket::threshold_of(rate).is_err(), "{rate:?}");
    }
}

/// Every request in exactly one batch, the bounds of RFC-0016 'Batching', each batch's keys in
/// order and its total its requests'.
fn check_bounds(policy: &AuditPolicy, requests: &[(RequestKey, u8, u64)], packed: &[Batch<u8>]) {
    let tokens: BTreeMap<RequestKey, (u8, u64)> =
        requests.iter().map(|(k, g, t)| (*k, (*g, *t))).collect();
    let mut seen: Vec<RequestKey> = packed.iter().flat_map(|b| b.requests.clone()).collect();
    seen.sort();
    let mut all: Vec<RequestKey> = tokens.keys().copied().collect();
    all.sort();
    assert_eq!(seen, all, "every request in exactly one batch");
    let mut small = BTreeMap::new();
    for b in packed {
        assert!(b.requests.windows(2).all(|w| w[0] < w[1]), "keys in order");
        assert!(
            b.requests.iter().all(|k| tokens[k].0 == b.group),
            "one group per batch"
        );
        let total: u64 = b.requests.iter().map(|k| tokens[k].1).sum();
        assert_eq!(b.tokens, total);
        assert!(
            b.tokens <= u64::from(policy.max_batch_tokens()) || b.requests.len() == 1,
            "a batch over the maximum"
        );
        if b.tokens < u64::from(policy.min_batch_tokens()) {
            *small.entry(b.group).or_insert(0) += 1;
        }
    }
    assert!(
        small.values().all(|&n| n <= 1),
        "two batches of a group under the minimum"
    );
    // Groups are packed in increasing order.
    assert!(packed.windows(2).all(|w| w[0].group <= w[1].group));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn batches_keep_their_bounds_whatever_the_order(
        min in 0u32..2000,
        extra in 0u32..3000,
        raw in prop::collection::vec((any::<[u8; 2]>(), 0u8..3, 0u64..6000), 0..60),
        rotate in 0usize..60,
    ) {
        let max = (2 * min + extra).max(1);
        let p = policy(Vec::new(), min, max);
        let mut seen = std::collections::BTreeSet::new();
        let mut requests = Vec::new();
        for (id, group, tokens) in raw {
            let key = RequestKey { session_id: [id[0]; 16], request_id: u64::from(id[1]) };
            if seen.insert(key) {
                requests.push((key, group, tokens));
            }
        }
        let packed = batches(&p, &requests).expect("distinct keys");
        check_bounds(&p, &requests, &packed);
        // The input's order changes nothing.
        let mut shuffled = requests.clone();
        if !shuffled.is_empty() {
            let r = rotate % shuffled.len();
            shuffled.rotate_left(r);
            shuffled.reverse();
        }
        prop_assert_eq!(batches(&p, &shuffled).expect("distinct keys"), packed);
    }
}

#[test]
fn batching_is_first_fit_decreasing() {
    let key = |id: u64| RequestKey {
        session_id: [0; 16],
        request_id: id,
    };
    let p = policy(Vec::new(), 1, 10);
    // Decreasing: 6 opens a batch, 5 does not fit it and opens a second, 4 fills the first and
    // 3 goes to the second.
    let requests = [
        (key(1), 0u8, 6),
        (key(2), 0, 5),
        (key(3), 0, 4),
        (key(4), 0, 3),
    ];
    let packed = batches(&p, &requests).expect("distinct keys");
    let shape: Vec<(Vec<u64>, u64)> = packed
        .iter()
        .map(|b| (b.requests.iter().map(|k| k.request_id).collect(), b.tokens))
        .collect();
    assert_eq!(shape, [(vec![1, 3], 10), (vec![2, 4], 8)]);
    // First fit, not best fit: under 20, 12 opens a batch and both 9s the second (18). The 2 fits
    // both and goes to the first, 14, though best fit would fill the second to 20.
    let wide = policy(Vec::new(), 1, 20);
    let requests = [
        (key(1), 0u8, 12),
        (key(2), 0, 9),
        (key(3), 0, 9),
        (key(4), 0, 2),
    ];
    let packed = batches(&wide, &requests).expect("distinct keys");
    assert_eq!(packed[0].requests, [key(1), key(4)]);
    assert_eq!(packed[1].requests, [key(2), key(3)]);
    // Ties in key order: the lower key goes first.
    let requests = [(key(9), 0u8, 6), (key(8), 0, 6), (key(7), 0, 4)];
    let packed = batches(&p, &requests).expect("distinct keys");
    assert_eq!(packed[0].requests, [key(7), key(8)]);
    assert_eq!(packed[1].requests, [key(9)]);
    // A request longer than the maximum is alone, and takes no other.
    let requests = [(key(1), 0u8, 11), (key(2), 0, 1)];
    let packed = batches(&p, &requests).expect("distinct keys");
    assert_eq!(packed.len(), 2);
    assert_eq!(packed[0].requests, [key(1)]);
    // A key given twice is refused.
    let requests = [(key(1), 0u8, 1), (key(1), 1, 2)];
    assert_eq!(batches(&p, &requests), Err(RepeatedRequest(key(1))));
}

#[test]
fn a_policy_is_valid_by_construction() {
    let params = || ParamSetId(Ascii::new("live-64/1").expect("ASCII"));
    let make =
        |buckets: Vec<RateBucket>, min, max| AuditPolicy::new(1, buckets, min, max, 2, params());
    let b = |max_tokens| RateBucket {
        max_tokens,
        threshold: 1,
    };
    for (buckets, min, max) in [
        (vec![b(10), b(10)], 1, 2),
        (vec![b(10), b(9)], 1, 2),
        (vec![b(10)], 0, 0),
        (vec![b(10)], 3, 5),
        (vec![b(10)], u32::MAX, u32::MAX),
    ] {
        let refused = make(buckets, min, max).expect_err("a policy RFC-0016 rules out");
        assert!(
            matches!(refused, ProtocolError::Policy { .. }),
            "{refused:?}"
        );
        assert_eq!(refused.code(), ErrorCode::Usage);
    }
    // At the edges: twice the minimum, no minimum, no buckets.
    for (buckets, min, max) in [
        (vec![b(10), b(11)], 3, 6),
        (vec![b(10)], 0, 1),
        (Vec::new(), u32::MAX / 2, u32::MAX - 1),
    ] {
        let p = make(buckets.clone(), min, max).expect("a valid policy");
        assert_eq!(p.buckets(), &buckets[..]);
        assert_eq!((p.min_batch_tokens(), p.max_batch_tokens()), (min, max));
        // Its encoding decodes back to it.
        assert_eq!(
            decode::<AuditPolicy>(&encode(&p), DecodeLimits::MESSAGE),
            Ok(p)
        );
    }
}

#[test]
fn the_prover_answers_one_commitment_per_manifest() {
    let (manifest, other_manifest) = ([1u8; 32], [2u8; 32]);
    let (c, other_c) = (coin_commitment(&[3; 32]), coin_commitment(&[4; 32]));
    let p = policy(vec![bucket(10, "0.5")], 1, 2);
    let other_p = policy(vec![bucket(10, "0.25")], 1, 2);
    // A first commitment gets a fresh share, never the manifest's: two fresh answers differ.
    let first = ShareRecord::answer(None, manifest, &p, c).expect("a fresh answer");
    let again = ShareRecord::answer(None, manifest, &p, c).expect("a fresh answer");
    assert_ne!(first.prover_share, again.prover_share);
    assert_ne!(first.prover_share, manifest);
    assert_eq!(
        (first.manifest_digest, first.verifier_commitment),
        (manifest, c)
    );
    assert_eq!(first.policy, p);
    // The same commitment and policy again: the same share.
    assert_eq!(
        ShareRecord::answer(Some(&first), manifest, &p, c),
        Ok(first.clone())
    );
    // Another commitment, or another policy, for the manifest: refused, ERR-PROTO-ORDER.
    for (policy, commitment) in [(&p, other_c), (&other_p, c)] {
        let refused = ShareRecord::answer(Some(&first), manifest, policy, commitment);
        assert_eq!(refused, Err(ProtocolError::Answered));
    }
    assert_eq!(ProtocolError::Answered.code(), ErrorCode::ProtoOrder);
    // A record of another manifest is the caller's mistake.
    assert!(matches!(
        ShareRecord::answer(Some(&first), other_manifest, &p, c),
        Err(ProtocolError::Internal { .. })
    ));
    // The record decodes back from its TCE.
    assert_eq!(
        decode::<ShareRecord>(&encode(&first), DecodeLimits::MESSAGE),
        Ok(first)
    );
}

#[test]
fn the_record_draws_the_only_sample_of_its_manifest() {
    let mut rng = Rng(0x0016_0007);
    let verifier_share = [5u8; 32];
    let manifest = rng.bytes();
    let p = policy(vec![bucket(u32::MAX, "0.5")], 1, 2);
    let record =
        ShareRecord::answer(None, manifest, &p, coin_commitment(&verifier_share)).expect("fresh");
    let candidates: Vec<(RequestKey, u64)> = (0..200).map(|_| (rng.key(), 7)).collect();
    let sample = record
        .draw(verifier_share, candidates.iter().copied())
        .expect("a reveal that opens");
    // The record's manifest, policy, commitment and share, with the reveal.
    assert_eq!(*sample.manifest_digest(), manifest);
    assert_eq!(sample.policy(), &p);
    assert_eq!(
        sample.coin_toss(),
        &CoinTossTranscript {
            verifier_commitment: record.verifier_commitment,
            prover_share: record.prover_share,
            verifier_share,
        }
    );
    let expected = AuditSample::draw(manifest, p, sample.coin_toss().clone(), candidates.clone())
        .expect("the same draw");
    assert_eq!(sample, expected);
    // A reveal that does not open the recorded commitment draws nothing.
    assert_eq!(
        record.draw([6; 32], candidates),
        Err(ProtocolError::CoinToss)
    );
}

#[test]
fn batching_at_its_edges() {
    let key = |id: u64| RequestKey {
        session_id: [0; 16],
        request_id: id,
    };
    let p = policy(Vec::new(), 1, 10);
    // A request of exactly the maximum fills its batch, and zero-token requests go where they
    // first fit: the first batch with room.
    let requests = [
        (key(1), 0u8, 10),
        (key(2), 0, 4),
        (key(3), 0, 0),
        (key(4), 0, 0),
    ];
    let packed = batches(&p, &requests).expect("distinct keys");
    let shape: Vec<(Vec<u64>, u64)> = packed
        .iter()
        .map(|b| (b.requests.iter().map(|k| k.request_id).collect(), b.tokens))
        .collect();
    assert_eq!(shape, [(vec![1, 3, 4], 10), (vec![2], 4)]);
    // A request of 2^64 − 1 tokens, a candidate with no length, is alone; the sum cannot wrap.
    let requests = [(key(1), 0u8, u64::MAX), (key(2), 0, 3), (key(3), 0, 0)];
    let packed = batches(&p, &requests).expect("distinct keys");
    assert_eq!(packed[0].requests, [key(1)]);
    assert_eq!(packed[0].tokens, u64::MAX);
    assert_eq!(packed[1].requests, [key(2), key(3)]);
}

// ----- The vectors ------------------------------------------------------------------------------

fn load() -> Value {
    let path = common::vectors().join("audit/audit.json");
    serde_json::from_str(&fs::read_to_string(path).expect("vector file")).expect("JSON")
}

fn hex<const N: usize>(v: &Value) -> [u8; N] {
    let s = v.as_str().expect("hex");
    assert_eq!(s.len(), 2 * N, "{s}");
    let mut out = [0u8; N];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex");
    }
    out
}

fn bytes(v: &Value) -> Vec<u8> {
    let s = v.as_str().expect("hex");
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex"))
        .collect()
}

fn key_of(v: &Value) -> RequestKey {
    RequestKey {
        session_id: hex(&v["session_id"]),
        request_id: v["request_id"].as_u64().expect("a request"),
    }
}

fn toss_of(v: &Value) -> CoinTossTranscript {
    CoinTossTranscript {
        verifier_commitment: hex(&v["verifier_commitment"]),
        prover_share: hex(&v["prover_share"]),
        verifier_share: hex(&v["verifier_share"]),
    }
}

fn policy_of(v: &Value) -> AuditPolicy {
    let n = |k: &str| v[k].as_u64().expect("an integer");
    AuditPolicy::new(
        n("period_ns"),
        v["buckets"]
            .as_array()
            .expect("buckets")
            .iter()
            .map(|b| RateBucket {
                max_tokens: b["max_tokens"].as_u64().expect("max") as u32,
                threshold: b["threshold"].as_u64().expect("threshold"),
            })
            .collect(),
        n("min_batch_tokens") as u32,
        n("max_batch_tokens") as u32,
        n("proof_deadline_ns"),
        ParamSetId(Ascii::new(v["params"].as_str().expect("params")).expect("ASCII")),
    )
    .expect("the oracle's policy is valid")
}

#[test]
fn the_oracles_rates_seeds_and_values() {
    let v = load();
    for case in v["rates"].as_array().expect("rates") {
        let rate = case["rate"].as_str().expect("a rate");
        assert_eq!(
            RateBucket::threshold_of(rate),
            Ok(case["threshold"].as_u64().expect("a threshold")),
            "{rate}"
        );
    }
    for rate in v["refused_rates"].as_array().expect("refused") {
        let rate = rate.as_str().expect("a rate");
        assert!(RateBucket::threshold_of(rate).is_err(), "{rate:?}");
    }
    for case in v["seeds"].as_array().expect("seeds") {
        let coin_toss = toss_of(&case["coin_toss"]);
        assert_eq!(coin_toss.check(), Ok(()));
        let seed = sample_seed(&coin_toss, &hex(&case["manifest_digest"]));
        assert_eq!(seed, hex::<32>(&case["sample_seed"]));
        for value in case["values"].as_array().expect("values") {
            assert_eq!(
                selection_value(&seed, &key_of(&value["key"])),
                value["value"].as_u64().expect("a value")
            );
        }
    }
}

#[test]
fn the_oracles_samples_encode_and_refusals_fail_where_it_says() {
    let v = load();
    for case in v["samples"].as_array().expect("samples") {
        let name = case["name"].as_str().expect("a name");
        let candidates = case["candidates"]
            .as_array()
            .expect("candidates")
            .iter()
            .map(|c| (key_of(&c["key"]), c["tokens"].as_u64().expect("tokens")));
        let sample = AuditSample::draw(
            hex(&case["manifest_digest"]),
            policy_of(&case["policy"]),
            toss_of(&case["coin_toss"]),
            candidates,
        )
        .expect("a coin toss that opens");
        assert_eq!(sample.seed(), hex::<32>(&case["sample_seed"]), "{name}");
        let expected: Vec<RequestKey> = case["selected"]
            .as_array()
            .expect("selected")
            .iter()
            .map(key_of)
            .collect();
        assert_eq!(sample.selected(), &expected[..], "{name}");
        let payload = bytes(&case["payload"]);
        assert_eq!(encode(&sample), payload, "{name}");
        assert_eq!(
            decode::<AuditSample>(&payload, DecodeLimits::MESSAGE),
            Ok(sample.clone())
        );
        for cut in 0..payload.len() {
            let error = decode::<AuditSample>(&payload[..cut], DecodeLimits::MESSAGE)
                .expect_err("a prefix");
            assert_eq!(error.code, ErrorCode::CodecTruncated, "{name}: {cut} bytes");
        }
        assert_eq!(
            envelope::seal(&schema::AUDIT_SAMPLE, &sample),
            bytes(&case["envelope"]),
            "{name}"
        );
    }
    for case in v["refused"].as_array().expect("refused") {
        let name = case["name"].as_str().expect("a name");
        let error =
            decode::<AuditSample>(&bytes(&case["payload"]), DecodeLimits::MESSAGE).expect_err(name);
        assert_eq!(
            error.code.as_str(),
            case["code"].as_str().expect("a code"),
            "{name}"
        );
        assert_eq!(
            error.offset,
            case["at"].as_u64().expect("an offset"),
            "{name}"
        );
    }
    for case in v["accepted"].as_array().expect("accepted") {
        let name = case["name"].as_str().expect("a name");
        let payload = bytes(&case["payload"]);
        let sample = decode::<AuditSample>(&payload, DecodeLimits::MESSAGE).expect(name);
        assert_eq!(encode(&sample), payload, "{name}");
    }
}

#[test]
fn the_oracles_batches() {
    let v = load();
    for case in v["batches"].as_array().expect("batches") {
        let name = case["name"].as_str().expect("a name");
        let p = policy(
            Vec::new(),
            case["min_batch_tokens"].as_u64().expect("min") as u32,
            case["max_batch_tokens"].as_u64().expect("max") as u32,
        );
        let requests: Vec<(RequestKey, u64, u64)> = case["requests"]
            .as_array()
            .expect("requests")
            .iter()
            .map(|r| {
                (
                    key_of(&r["key"]),
                    r["group"].as_u64().expect("a group"),
                    r["tokens"].as_u64().expect("tokens"),
                )
            })
            .collect();
        let packed = batches(&p, &requests).expect("distinct keys");
        let expected: Vec<Batch<u64>> = case["batches"]
            .as_array()
            .expect("batches")
            .iter()
            .map(|b| Batch {
                group: b["group"].as_u64().expect("a group"),
                requests: b["requests"]
                    .as_array()
                    .expect("keys")
                    .iter()
                    .map(key_of)
                    .collect(),
                tokens: b["tokens"].as_u64().expect("tokens"),
            })
            .collect();
        assert_eq!(packed, expected, "{name}");
    }
}
