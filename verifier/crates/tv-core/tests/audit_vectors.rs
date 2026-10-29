// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The audit's accounting against `spec/vectors/audit/`, written by an independent oracle
//! (`generate.py`): the rates' thresholds and the texts that are no rate; each coin toss's seed
//! and its keys' selection values; each sample's selection from its candidates, its payload, its
//! every proper prefix and its envelope; the payloads a reader refuses, with code and offset, and
//! those at the rules' edges it accepts; and the batches first-fit decreasing packs. The
//! candidates of the two-period logs are `anchor_vectors.rs`'s.

use tv_codec::{DecodeLimits, ErrorCode};
use tv_core::anchor::Candidate;
use tv_core::audit::{
    AuditPolicy, AuditSample, BatchRequest, CoinTossTranscript, RateBucket, RequestKey, pack,
    select, selection_value,
};
use tv_core::claims::ParamSetId;
use tv_hash::{Domain, hash};
use tv_testkit::{Json, hex, read_vectors};

fn text<'a>(value: &'a Json, key: &str) -> &'a str {
    value
        .get(key)
        .and_then(Json::as_str)
        .unwrap_or_else(|| panic!("`{key}`"))
}
fn number(value: &Json, key: &str) -> u64 {
    value
        .get(key)
        .and_then(Json::as_u64)
        .unwrap_or_else(|| panic!("`{key}`"))
}
fn list<'a>(value: &'a Json, key: &str) -> &'a [Json] {
    value
        .get(key)
        .and_then(Json::as_array)
        .unwrap_or_else(|| panic!("`{key}`"))
}
fn bytes32(text: &str) -> [u8; 32] {
    hex(text).try_into().unwrap()
}
fn key_of(value: &Json) -> RequestKey {
    RequestKey {
        session_id: hex(text(value, "session_id")).try_into().unwrap(),
        request_id: number(value, "request_id"),
    }
}
fn toss_of(value: &Json) -> CoinTossTranscript {
    CoinTossTranscript {
        verifier_commitment: bytes32(text(value, "verifier_commitment")),
        prover_share: bytes32(text(value, "prover_share")),
        verifier_share: bytes32(text(value, "verifier_share")),
    }
}
fn policy_of(value: &Json) -> AuditPolicy {
    AuditPolicy::new(
        number(value, "period_ns"),
        list(value, "buckets")
            .iter()
            .map(|b| RateBucket {
                max_tokens: number(b, "max_tokens") as u32,
                threshold: number(b, "threshold"),
            })
            .collect(),
        number(value, "min_batch_tokens") as u32,
        number(value, "max_batch_tokens") as u32,
        number(value, "proof_deadline_ns"),
        ParamSetId::new(text(value, "params")).unwrap(),
    )
    .unwrap()
}

#[test]
fn every_rate_has_the_oracles_threshold_and_every_other_text_none() {
    let doc = read_vectors("audit/audit.json");
    let rates = list(&doc, "rates");
    assert_eq!(rates.len(), 12);
    for r in rates {
        assert_eq!(
            RateBucket::threshold_of(text(r, "rate")),
            Some(number(r, "threshold")),
            "{}",
            text(r, "rate")
        );
    }
    let refused = list(&doc, "refused_rates");
    assert_eq!(refused.len(), 13);
    for r in refused {
        let r = r.as_str().unwrap();
        assert_eq!(RateBucket::threshold_of(r), None, "{r:?}");
    }
}

#[test]
fn every_toss_opens_and_gives_the_oracles_seed_and_selection_values() {
    let doc = read_vectors("audit/audit.json");
    let seeds = list(&doc, "seeds");
    assert_eq!(seeds.len(), 4);
    let mut values = 0;
    for case in seeds {
        let toss = toss_of(case.get("coin_toss").unwrap());
        toss.check().unwrap();
        let manifest = bytes32(text(case, "manifest_digest"));
        let seed = toss.sample_seed(&manifest);
        assert_eq!(seed, bytes32(text(case, "sample_seed")));
        for v in list(case, "values") {
            let key = key_of(v.get("key").unwrap());
            assert_eq!(selection_value(&seed, &key), number(v, "value"));
            values += 1;
        }
    }
    assert_eq!(values, 32);
}

#[test]
fn every_sample_is_selected_encoded_cut_and_sealed_as_the_oracle_writes_it() {
    let doc = read_vectors("audit/audit.json");
    let samples = list(&doc, "samples");
    assert_eq!(samples.len(), 4);
    let (mut past_every_bucket, mut selected_total) = (0, 0);
    for case in samples {
        let name = text(case, "name");
        let policy = policy_of(case.get("policy").unwrap());
        let toss = toss_of(case.get("coin_toss").unwrap());
        let manifest = bytes32(text(case, "manifest_digest"));
        let seed = toss.sample_seed(&manifest);
        assert_eq!(seed, bytes32(text(case, "sample_seed")), "{name}");

        // The selection, from the candidates and their lengths.
        let candidates: Vec<Candidate> = list(case, "candidates")
            .iter()
            .map(|c| {
                let key = key_of(c.get("key").unwrap());
                Candidate {
                    session_id: key.session_id,
                    request_id: key.request_id,
                    ended: true,
                    tokens: number(c, "tokens"),
                }
            })
            .collect();
        past_every_bucket += candidates.iter().filter(|c| c.tokens > 131_072).count();
        let expected: Vec<RequestKey> = list(case, "selected").iter().map(key_of).collect();
        selected_total += expected.len();
        assert_eq!(
            select(&policy, &seed, &candidates).unwrap(),
            expected,
            "{name}"
        );

        // The payload decodes to the case's fields and encodes back to itself.
        let payload = hex(text(case, "payload"));
        let sample = AuditSample::decode_payload(&payload, &DecodeLimits::MESSAGE).unwrap();
        assert_eq!(sample.manifest_digest(), &manifest, "{name}");
        assert_eq!(sample.policy(), &policy, "{name}");
        assert_eq!(sample.coin_toss(), &toss, "{name}");
        assert_eq!(sample.selected(), expected.as_slice(), "{name}");
        assert_eq!(sample.seed(), seed, "{name}");
        assert_eq!(sample.encode_payload().unwrap(), payload, "{name}");

        // Every proper prefix is cut short, at the oracle's offsets.
        let mut offsets = std::collections::BTreeSet::new();
        for cut in 0..payload.len() {
            let error = AuditSample::decode_payload(&payload[..cut], &DecodeLimits::MESSAGE)
                .expect_err("a proper prefix decodes");
            assert_eq!(error.code(), ErrorCode::Truncated, "{name}: {cut}");
            offsets.insert(error.offset());
        }
        let oracle: std::collections::BTreeSet<u64> = list(case, "truncated_at")
            .iter()
            .map(|n| n.as_u64().unwrap())
            .collect();
        assert_eq!(offsets, oracle, "{name}");

        // The envelope: sealed byte for byte, opened to the same sample, its identity its tail.
        let envelope = hex(text(case, "envelope"));
        assert_eq!(sample.seal().unwrap(), envelope, "{name}");
        let opened = AuditSample::open(&envelope, &DecodeLimits::MESSAGE).unwrap();
        assert_eq!(opened, sample, "{name}");
        assert_eq!(
            &envelope[envelope.len() - 32..],
            opened.identity(),
            "{name}"
        );
        assert_eq!(
            opened.identity(),
            &hash(Domain::Envelope, &envelope[..envelope.len() - 32])
        );
    }
    // The cases cover candidates past every bucket, always selected, and non-empty samples.
    assert!(past_every_bucket > 0 && selected_total > 0);
}

#[test]
fn every_refused_payload_is_refused_with_the_oracles_code_and_offset() {
    let doc = read_vectors("audit/audit.json");
    let refused = list(&doc, "refused");
    assert_eq!(refused.len(), 8);
    for case in refused {
        let name = text(case, "name");
        let payload = hex(text(case, "payload"));
        let error = AuditSample::decode_payload(&payload, &DecodeLimits::MESSAGE)
            .expect_err("a refused payload decodes");
        let code = match text(case, "code") {
            "ERR-CODEC-NONCANONICAL" => ErrorCode::Noncanonical,
            "ERR-CODEC-LIMIT" => ErrorCode::Limit,
            "ERR-CODEC-TRUNCATED" => ErrorCode::Truncated,
            other => panic!("{other}"),
        };
        assert_eq!(
            (error.code(), error.offset()),
            (code, number(case, "at")),
            "{name}"
        );
    }
    let accepted = list(&doc, "accepted");
    assert_eq!(accepted.len(), 3);
    for case in accepted {
        let payload = hex(text(case, "payload"));
        let sample = AuditSample::decode_payload(&payload, &DecodeLimits::MESSAGE)
            .unwrap_or_else(|e| panic!("{}: {e:?}", text(case, "name")));
        assert_eq!(sample.encode_payload().unwrap(), payload);
    }
}

#[test]
fn every_batch_set_is_packed_as_the_oracle_packs_it() {
    let doc = read_vectors("audit/audit.json");
    let cases = list(&doc, "batches");
    assert_eq!(cases.len(), 6);
    for case in cases {
        let name = text(case, "name");
        let requests: Vec<BatchRequest<u64>> = list(case, "requests")
            .iter()
            .map(|r| BatchRequest {
                key: key_of(r.get("key").unwrap()),
                group: number(r, "group"),
                tokens: number(r, "tokens"),
            })
            .collect();
        let max = number(case, "max_batch_tokens") as u32;
        let min = number(case, "min_batch_tokens");
        let packed = pack(max, &requests).unwrap();
        let expected = list(case, "batches");
        assert_eq!(packed.len(), expected.len(), "{name}");
        for (got, want) in packed.iter().zip(expected) {
            assert_eq!(got.group, number(want, "group"), "{name}");
            assert_eq!(got.tokens, number(want, "tokens"), "{name}");
            let keys: Vec<RequestKey> = list(want, "requests").iter().map(key_of).collect();
            assert_eq!(got.requests, keys, "{name}");
        }
        // RFC-0016 + RFC-0071 bounds: per (group, slot_class), at most one batch has counted
        // tokens below min_batch_tokens, and every multi-request batch has counted tokens <= max.
        let class_of = |key: RequestKey| -> u32 {
            let t = requests.iter().find(|r| r.key == key).unwrap().tokens;
            u64::BITS.saturating_sub(t.saturating_sub(1).leading_zeros())
        };
        let counted_of = |b: &tv_core::audit::Batch<u64>| -> u128 {
            let c = class_of(b.requests[0]);
            for &k in &b.requests {
                assert_eq!(class_of(k), c, "{name}: mixed classes in batch");
            }
            let r_bits =
                usize::BITS.saturating_sub(b.requests.len().saturating_sub(1).leading_zeros());
            1u128 << (c + r_bits)
        };
        let mut group_classes: Vec<(u64, u32)> = packed
            .iter()
            .map(|b| (b.group, class_of(b.requests[0])))
            .collect();
        group_classes.dedup();
        for (g, c) in group_classes {
            let small = packed
                .iter()
                .filter(|b| {
                    b.group == g && class_of(b.requests[0]) == c && counted_of(b) < u128::from(min)
                })
                .count();
            assert!(small <= 1, "{name}: group {g} class {c}");
        }
        for b in &packed {
            assert!(
                counted_of(b) <= u128::from(max) || b.requests.len() == 1,
                "{name}"
            );
        }
    }
}
