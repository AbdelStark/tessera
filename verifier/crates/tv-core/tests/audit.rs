// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The audit's accounting, rule by rule: each test breaks one rule of RFC-0016 'Policy', 'Sample
//! derivation', 'Verifier checks' and 'Audit report', or RFC-0019 'Ledger', and asserts the exact
//! verdict or error; seeded cases from `tv-testkit` check the selection against a reference
//! written in the test, with the test kit's own BLAKE3.

mod anchor_support;

use anchor_support::{GENESIS_NS, PERIOD_NS, Scenario, chain, end, inbound, open, outbound};
use std::fs;
use std::path::PathBuf;
use tv_codec::{DecodeLimits, ErrorCode};
use tv_core::anchor::{Candidate, Failure, VerifiedLog, Window};
use tv_core::audit::{
    AuditPolicy, AuditReport, AuditSample, AuditToss, BitSet, CoinTossTranscript, LedgerFile,
    PolicyError, RateBucket, RequestKey, RequestOutcome, UnprovenReason, VerifierLedger,
    read_ledger, select, selection_value,
};
use tv_core::claims::ParamSetId;
use tv_core::protocol::{BatchDescriptor, FixedBits, PublicStatement, RequestRef};
use tv_core::validation::StatementField;
use tv_core::verdict::{AnchorFailure, ErrorCode as Code, RejectReason, VerifierError};
use tv_hash::{Domain, hash};
use tv_testkit::{Json, SplitMix64, blake3, read_vectors};

// ----- Helpers ---------------------------------------------------------------------------------

fn policy(buckets: &[(u32, u64)]) -> AuditPolicy {
    AuditPolicy::new(
        86_400_000_000_000,
        buckets
            .iter()
            .map(|&(max_tokens, threshold)| RateBucket {
                max_tokens,
                threshold,
            })
            .collect(),
        16_384,
        262_144,
        604_800_000_000_000,
        ParamSetId::new("live-64/1").unwrap(),
    )
    .unwrap()
}

/// Every request of up to 4,096 tokens at rate 1/2, of up to 32,768 at 1/4.
fn halves() -> AuditPolicy {
    policy(&[(4096, 1 << 63), (32_768, 1 << 62)])
}

/// Selects everything: rate 1 to 2^32 tokens.
fn everything() -> AuditPolicy {
    policy(&[(u32::MAX, u64::MAX)])
}

fn toss(s_v: [u8; 32], s_p: [u8; 32]) -> CoinTossTranscript {
    CoinTossTranscript {
        verifier_commitment: hash(Domain::CoinCommit, &s_v),
        prover_share: s_p,
        verifier_share: s_v,
    }
}

fn key(s: u8, r: u64) -> RequestKey {
    RequestKey {
        session_id: [s; 16],
        request_id: r,
    }
}

fn req(s: u8, r: u64) -> RequestRef {
    RequestRef {
        request_id: r,
        session_id: [s; 16],
        prompt_len: 1,
        output_len: 1,
    }
}

fn statement(manifest: [u8; 32], requests: Vec<RequestRef>) -> PublicStatement {
    PublicStatement {
        claim_list_digest: [0; 32],
        weights_root: [0; 32],
        weights_epoch: 0,
        manifest_digest: manifest,
        batch: BatchDescriptor { requests },
        predictor: anchor_support::greedy(),
    }
}

fn batch_mismatch() -> RejectReason {
    RejectReason::StatementMismatch {
        field: StatementField::Batch,
    }
}

fn batch_rejected() -> Failure {
    Failure::Rejected(batch_mismatch())
}

fn inconclusive(code: Code) -> Failure {
    Failure::Inconclusive(VerifierError { code })
}

/// A log of session 1's requests 1 to 5, each of one inbound and one outbound frame, ended.
fn five_requests() -> Scenario {
    let mut items = vec![open(1)];
    for r in 1..=5 {
        items.push(inbound(1, r, 0, 2).item());
        items.push(outbound(1, r, 0, 3).item());
        items.push(end(1, r));
    }
    Scenario::one(items)
}

/// A sample selecting session 1's requests 1 to 5.
fn sample_of_five() -> (Scenario, AuditSample) {
    let scenario = five_requests();
    let sample =
        AuditSample::derive(&scenario.window(), everything(), toss([1; 32], [2; 32])).unwrap();
    assert_eq!(sample.selected().len(), 5);
    (scenario, sample)
}

/// A sample with `keys` selected, decoded from a payload built here: the manifest `[7; 32]`, the
/// policy that selects everything, and a toss that opens.
fn sample_with(keys: &[RequestKey]) -> AuditSample {
    let mut payload = vec![7; 32];
    // period, one bucket, batch bounds, deadline, name.
    payload.extend(86_400u64.to_le_bytes());
    payload.extend(1u64.to_le_bytes());
    payload.extend(u32::MAX.to_le_bytes());
    payload.extend(u64::MAX.to_le_bytes());
    payload.extend(16_384u32.to_le_bytes());
    payload.extend(262_144u32.to_le_bytes());
    payload.extend(604_800u64.to_le_bytes());
    payload.extend(9u64.to_le_bytes());
    payload.extend(b"live-64/1");
    let t = toss([3; 32], [4; 32]);
    payload.extend(t.verifier_commitment);
    payload.extend(t.prover_share);
    payload.extend(t.verifier_share);
    payload.extend((keys.len() as u64).to_le_bytes());
    for k in keys {
        payload.extend(k.session_id);
        payload.extend(k.request_id.to_le_bytes());
    }
    AuditSample::decode_payload(&payload, &DecodeLimits::MESSAGE).unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let mut rng = SplitMix64::new(std::process::id().into());
    let dir = std::env::temp_dir().join(format!(
        "tessera-audit-{name}-{}-{:x}",
        std::process::id(),
        rng.next_u64() ^ name.len() as u64
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

// ----- The policy ------------------------------------------------------------------------------

#[test]
fn a_policy_refuses_each_rule_it_breaks_and_takes_its_edges() {
    let make = |buckets: &[(u32, u64)], min: u32, max: u32| {
        AuditPolicy::new(
            0,
            buckets
                .iter()
                .map(|&(max_tokens, threshold)| RateBucket {
                    max_tokens,
                    threshold,
                })
                .collect(),
            min,
            max,
            0,
            ParamSetId::new("live-64/1").unwrap(),
        )
        .map(|_| ())
    };
    // Buckets not strictly increasing, named by the first that breaks the rule.
    assert_eq!(
        make(&[(10, 1), (10, 2)], 1, 2),
        Err(PolicyError::BucketOrder { index: 1 })
    );
    assert_eq!(
        make(&[(10, 1), (20, 2), (19, 2)], 1, 2),
        Err(PolicyError::BucketOrder { index: 2 })
    );
    // The batch bounds: 0, and below twice the minimum, the doubling past u32 included.
    assert_eq!(make(&[], 0, 0), Err(PolicyError::BatchBounds));
    assert_eq!(make(&[], 16_385, 32_769), Err(PolicyError::BatchBounds));
    assert_eq!(make(&[], u32::MAX, u32::MAX), Err(PolicyError::BatchBounds));
    // The order of the checks: the buckets first.
    assert_eq!(
        make(&[(5, 0), (5, 0)], 0, 0),
        Err(PolicyError::BucketOrder { index: 1 })
    );
    // The edges pass.
    assert_eq!(make(&[], 16_384, 32_768), Ok(()));
    assert_eq!(make(&[], 0, 1), Ok(()));
    assert_eq!(make(&[(0, 0), (u32::MAX, u64::MAX)], 0, 1), Ok(()));
}

#[test]
fn a_bucket_is_the_first_whose_bound_holds_the_length_and_past_them_all_selects() {
    let p = policy(&[(10, 0), (20, 5), (30, u64::MAX)]);
    for (tokens, threshold) in [
        (0, Some(0)),
        (10, Some(0)),
        (11, Some(5)),
        (20, Some(5)),
        (21, Some(u64::MAX)),
        (30, Some(u64::MAX)),
        (31, None),
        (u64::MAX, None),
    ] {
        assert_eq!(p.bucket_threshold(tokens), threshold, "{tokens}");
    }
    // Rate 0 selects nothing, a length past every bucket everything, whatever the value.
    for value in [0, 4, 5, u64::MAX - 1, u64::MAX] {
        assert!(!p.selects(value, 10));
        assert_eq!(p.selects(value, 15), value < 5);
        assert!(p.selects(value, 31));
    }
    // With no bucket, every length is past them all.
    assert!(policy(&[]).selects(u64::MAX, 0));
}

// ----- The sample ------------------------------------------------------------------------------

#[test]
fn the_sample_is_derived_from_the_windows_candidates_over_both_periods() {
    // P − 1 holds request 9's first frames, unterminated there; P its end and request 1 to 3;
    // request 4 is a frame of impossible length; request 5 has frames in P − 1 and no end.
    let scenario = Scenario::two(
        vec![
            open(1),
            inbound(1, 9, 0, 5).item(),
            inbound(1, 5, 0, 1).item(),
        ],
        vec![
            anchor_support::carry(1),
            outbound(1, 9, 0, 6).item(),
            end(1, 9),
            inbound(1, 1, 0, 1).item(),
            end(1, 1),
            inbound(1, 2, 0, 5000).item(),
            end(1, 2),
            inbound(1, 3, 0, 40_000).item(),
            end(1, 3),
            inbound(1, 4, 0, 1).length(23),
            end(1, 4),
        ],
    );
    let window = scenario.window();
    let candidates = window.candidates().unwrap();
    assert_eq!(candidates.len(), 6);
    let t = toss([5; 32], [6; 32]);
    let sample = AuditSample::derive(&window, halves(), t).unwrap();
    let seed = t.sample_seed(scenario.period.manifest_digest());
    assert_eq!(sample.seed(), seed);
    assert_eq!(sample.manifest_digest(), scenario.period.manifest_digest());
    assert_eq!(
        sample.selected(),
        select(&halves(), &seed, &candidates).unwrap()
    );
    // 40,000 tokens and an impossible length are past every bucket: selected.
    assert!(sample.index_of(&key(1, 3)).is_some());
    assert!(sample.index_of(&key(1, 4)).is_some());
    // Each decision is the value against the bucket's threshold.
    for c in &candidates {
        let k = RequestKey {
            session_id: c.session_id,
            request_id: c.request_id,
        };
        let expected = halves().selects(selection_value(&seed, &k), c.tokens);
        assert_eq!(sample.index_of(&k).is_some(), expected, "{k:?}");
    }
    sample.check_derivation(&window).unwrap();
    // The sealed sample opens to itself, and its identity is the envelope's tail.
    let tas = sample.seal().unwrap();
    let opened = AuditSample::open(&tas, &DecodeLimits::MESSAGE).unwrap();
    assert_eq!(opened, sample);
    assert_eq!(opened.identity(), &tas[tas.len() - 32..]);
}

#[test]
fn a_period_of_zero_length_selects_nothing() {
    let built = chain(&[
        (
            GENESIS_NS,
            GENESIS_NS + PERIOD_NS,
            vec![open(1), inbound(1, 1, 0, 1).item()],
        ),
        (GENESIS_NS + PERIOD_NS, GENESIS_NS + PERIOD_NS, vec![]),
    ]);
    let logs: Vec<VerifiedLog> = built.iter().map(|b| b.verified()).collect();
    let window = Window::bind(&logs[1], &[], Some(&logs[0])).unwrap();
    // Request 1 is unterminated in P − 1, but a zero-length period draws from nothing.
    let sample = AuditSample::derive(&window, everything(), toss([1; 32], [2; 32])).unwrap();
    assert!(sample.selected().is_empty());
}

#[test]
fn a_coin_toss_reveal_that_does_not_open_is_refused() {
    let scenario = five_requests();
    let mut t = toss([1; 32], [2; 32]);
    t.verifier_share[31] ^= 1;
    assert_eq!(
        AuditSample::derive(&scenario.window(), everything(), t),
        Err(inconclusive(Code::ProtocolCoinToss))
    );
    assert_eq!(
        t.check(),
        Err(VerifierError {
            code: Code::ProtocolCoinToss
        })
    );
    // A decoded sample whose toss does not open has no report.
    let sample = sample_with(&[key(1, 1)]);
    let mut payload = sample.encode_payload().unwrap();
    let commitment_at = payload.len() - 8 - 24 - 96;
    payload[commitment_at] ^= 1;
    let broken = AuditSample::decode_payload(&payload, &DecodeLimits::MESSAGE).unwrap();
    assert_eq!(
        AuditReport::from_ledger(&broken, &VerifierLedger::new()),
        Err(inconclusive(Code::ProtocolCoinToss))
    );
}

#[test]
fn a_sample_that_is_not_the_verifiers_derivation_is_refused() {
    let (scenario, sample) = sample_of_five();
    let window = scenario.window();
    sample.check_derivation(&window).unwrap();
    // One selected request dropped from the payload.
    let mut payload = sample.encode_payload().unwrap();
    let count_at = payload.len() - 8 - 5 * 24;
    payload[count_at] = 4;
    payload.truncate(payload.len() - 24);
    let fewer = AuditSample::decode_payload(&payload, &DecodeLimits::MESSAGE).unwrap();
    assert_eq!(
        fewer.check_derivation(&window),
        Err(inconclusive(Code::Usage))
    );
    // Another manifest.
    let other = sample_with(sample.selected());
    assert_eq!(
        other.check_derivation(&window),
        Err(inconclusive(Code::Usage))
    );
}

#[test]
fn the_verifiers_toss_commits_a_fresh_share_and_reveals_only_over_its_manifest() {
    let built = chain(&[
        (
            GENESIS_NS,
            GENESIS_NS + PERIOD_NS,
            vec![open(1), inbound(1, 1, 0, 1).item(), end(1, 1)],
        ),
        (
            GENESIS_NS + PERIOD_NS,
            GENESIS_NS + 2 * PERIOD_NS,
            vec![
                anchor_support::carry(1),
                inbound(1, 2, 0, 1).item(),
                end(1, 2),
            ],
        ),
    ]);
    let logs: Vec<VerifiedLog> = built.iter().map(|b| b.verified()).collect();
    let first = Window::bind(&logs[0], &[], None).unwrap();
    let second = Window::bind(&logs[1], &[], Some(&logs[0])).unwrap();
    let a = AuditToss::commit(&built[1].checked(), everything()).unwrap();
    let b = AuditToss::commit(&built[1].checked(), everything()).unwrap();
    assert_ne!(a.commitment(), b.commitment(), "a fresh share per toss");
    assert_eq!(a.manifest_digest(), &built[1].identity);
    // The reveal over the other period's window is the verifier's own inputs disagreeing, and
    // leaves the toss usable.
    assert_eq!(
        a.reveal(&first, [9; 32]).err(),
        Some(inconclusive(Code::Usage))
    );
    let sample = a.reveal(&second, [9; 32]).unwrap();
    let t = sample.coin_toss();
    assert_eq!(&t.verifier_commitment, a.commitment());
    assert_eq!(hash(Domain::CoinCommit, &t.verifier_share), *a.commitment());
    assert_eq!(t.prover_share, [9; 32]);
    assert_eq!(sample.selected(), [key(1, 2)]);
    sample.check_derivation(&second).unwrap();
}

// ----- The accounting checks -------------------------------------------------------------------

#[test]
fn a_batch_of_selected_requests_passes_and_is_recorded_with_its_uhat() {
    let (scenario, sample) = sample_of_five();
    let manifest = *scenario.period.manifest_digest();
    let mut ledger = VerifierLedger::new();
    let s = statement(manifest, vec![req(1, 4), req(1, 2)]);
    assert_eq!(ledger.check_accounting(&sample, &s), Ok(vec![3, 1]));
    ledger
        .record_accepted(&sample, &s, &[FixedBits(40), FixedBits(20)])
        .unwrap();
    let record = ledger.proven(sample.identity()).unwrap();
    assert_eq!(
        record.iter().collect::<Vec<_>>(),
        [(1, FixedBits(20)), (3, FixedBits(40))]
    );
    // A second batch merges by index.
    let s2 = statement(manifest, vec![req(1, 5), req(1, 1)]);
    ledger
        .record_accepted(&sample, &s2, &[FixedBits(50), FixedBits(10)])
        .unwrap();
    let record = ledger.proven(sample.identity()).unwrap();
    assert_eq!(
        record.iter().collect::<Vec<_>>(),
        [
            (0, FixedBits(10)),
            (1, FixedBits(20)),
            (3, FixedBits(40)),
            (4, FixedBits(50))
        ]
    );
    // The ledger round-trips through its artifact.
    let tvl = ledger.seal().unwrap();
    assert_eq!(
        VerifierLedger::open(&tvl, &DecodeLimits::MESSAGE).unwrap(),
        ledger
    );
}

#[test]
fn a_request_not_in_the_sample_is_rejected() {
    let (scenario, sample) = sample_of_five();
    let manifest = *scenario.period.manifest_digest();
    let ledger = VerifierLedger::new();
    // Another request id, another session, and a selected request beside an unselected one.
    for requests in [vec![req(1, 6)], vec![req(2, 1)], vec![req(1, 1), req(1, 0)]] {
        assert_eq!(
            ledger.check_accounting(&sample, &statement(manifest, requests)),
            Err(batch_rejected())
        );
    }
}

#[test]
fn a_request_in_two_accepted_batches_is_rejected_the_second_time() {
    let (scenario, sample) = sample_of_five();
    let manifest = *scenario.period.manifest_digest();
    let mut ledger = VerifierLedger::new();
    let first = statement(manifest, vec![req(1, 1), req(1, 2)]);
    ledger
        .record_accepted(&sample, &first, &[FixedBits(0), FixedBits(0)])
        .unwrap();
    let before = ledger.clone();
    // The replay, alone or beside a new request.
    for requests in [vec![req(1, 2)], vec![req(1, 3), req(1, 1)]] {
        let s = statement(manifest, requests);
        assert_eq!(ledger.check_accounting(&sample, &s), Err(batch_rejected()));
        assert_eq!(
            ledger.record_accepted(&sample, &s, &[FixedBits(0); 2][..s.batch.requests.len()]),
            Err(Failure::Rejected(batch_mismatch()))
        );
        assert_eq!(ledger, before, "a refused record changes nothing");
    }
    // A request named twice in one batch, next to each other or apart.
    for requests in [
        vec![req(1, 4), req(1, 4)],
        vec![req(1, 5), req(1, 3), req(1, 4), req(1, 5)],
    ] {
        assert_eq!(
            ledger.check_accounting(&sample, &statement(manifest, requests)),
            Err(batch_rejected())
        );
    }
    // Another sample's record does not count against this one.
    let other = sample_with(sample.selected());
    let s = statement([7; 32], vec![req(1, 1)]);
    assert_eq!(ledger.check_accounting(&other, &s), Ok(vec![0]));
}

#[test]
fn another_manifest_is_rejected_before_the_batch() {
    let (scenario, sample) = sample_of_five();
    let ledger = VerifierLedger::new();
    let mut manifest = *scenario.period.manifest_digest();
    manifest[0] ^= 1;
    // The batch is broken too; the manifest's field comes first (RFC-0039 'The checks').
    for requests in [vec![req(1, 1)], vec![req(9, 9)]] {
        assert_eq!(
            ledger.check_accounting(&sample, &statement(manifest, requests)),
            Err(Failure::Rejected(RejectReason::StatementMismatch {
                field: StatementField::ManifestDigest
            }))
        );
    }
}

#[test]
fn a_record_whose_uhat_does_not_fit_the_batch_changes_nothing() {
    let (scenario, sample) = sample_of_five();
    let manifest = *scenario.period.manifest_digest();
    let mut ledger = VerifierLedger::new();
    let s = statement(manifest, vec![req(1, 1), req(1, 2)]);
    for uhat in [&[][..], &[FixedBits(1)], &[FixedBits(1); 3]] {
        assert_eq!(
            ledger.record_accepted(&sample, &s, uhat),
            Err(inconclusive(Code::Usage))
        );
        assert_eq!(ledger, VerifierLedger::new());
    }
}

// ----- The ledger's encoding -------------------------------------------------------------------

/// A ledger payload with no weight entries and the `proven` records given as raw bytes.
fn ledger_payload(proven: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for _ in 0..3 {
        out.extend(0u64.to_le_bytes());
    }
    out.extend((proven.len() as u64).to_le_bytes());
    for p in proven {
        out.extend(p);
    }
    out
}

/// One `proven` record: the sample, the set's bytes and the `Û` list.
fn record(sample: u8, set: &[u8], uhat: &[u64]) -> Vec<u8> {
    let mut out = vec![sample; 32];
    out.extend((set.len() as u64).to_le_bytes());
    out.extend(set);
    out.extend((uhat.len() as u64).to_le_bytes());
    for u in uhat {
        out.extend(u.to_le_bytes());
    }
    out
}

fn refusal(payload: &[u8]) -> (ErrorCode, u64) {
    let e = VerifierLedger::decode_payload(payload, &DecodeLimits::MESSAGE).unwrap_err();
    (e.code(), e.offset())
}

#[test]
fn a_ledger_that_breaks_an_encoding_rule_does_not_decode() {
    // `proven`'s length is at 24, after three empty sequences.
    let first = 24;
    // A record decodes; its set holds indices 0 and 9.
    let good = ledger_payload(&[record(1, &[0x01, 0x02], &[5, 6])]);
    let ledger = VerifierLedger::decode_payload(&good, &DecodeLimits::MESSAGE).unwrap();
    let r = &ledger.proven_samples()[0];
    assert_eq!(
        r.iter().collect::<Vec<_>>(),
        [(0, FixedBits(5)), (9, FixedBits(6))]
    );
    assert_eq!(ledger.encode_payload().unwrap(), good);
    // A set whose last byte is zero, at that byte.
    let set_at = first + 8 + 32;
    assert_eq!(
        refusal(&ledger_payload(&[record(1, &[0x01, 0x00], &[5])])),
        (ErrorCode::Noncanonical, set_at as u64 + 8 + 1)
    );
    // An empty set, at its length.
    assert_eq!(
        refusal(&ledger_payload(&[record(1, &[], &[])])),
        (ErrorCode::Noncanonical, set_at as u64)
    );
    // A Û list of another length than the set, at the list's length.
    let uhat_at = set_at + 8 + 2;
    for uhat in [&[5][..], &[5, 6, 7]] {
        assert_eq!(
            refusal(&ledger_payload(&[record(1, &[0x01, 0x02], uhat)])),
            (ErrorCode::Noncanonical, uhat_at as u64)
        );
    }
    // Samples not strictly increasing, at the second's first byte.
    let one = record(2, &[0x01], &[0]);
    let second_at = first + 8 + one.len();
    for later in [2, 1] {
        assert_eq!(
            refusal(&ledger_payload(&[
                one.clone(),
                record(later, &[0x01], &[0])
            ])),
            (ErrorCode::Noncanonical, second_at as u64),
            "{later}"
        );
    }
    // A blinding-row set whose last byte is zero.
    let mut rows = Vec::new();
    rows.extend(0u64.to_le_bytes());
    rows.extend(0u64.to_le_bytes());
    rows.extend(1u64.to_le_bytes());
    rows.extend([4; 32]);
    rows.extend(3u32.to_le_bytes());
    rows.push(2);
    rows.extend(1u64.to_le_bytes());
    rows.push(0);
    rows.extend(0u64.to_le_bytes());
    assert_eq!(refusal(&rows), (ErrorCode::Noncanonical, 16 + 8 + 37 + 8));
    // Every proper prefix is cut short; a byte past the value is left over.
    for cut in 0..good.len() {
        assert_eq!(refusal(&good[..cut]).0, ErrorCode::Truncated, "{cut}");
    }
    let mut longer = good.clone();
    longer.push(0);
    assert_eq!(
        refusal(&longer),
        (ErrorCode::Noncanonical, good.len() as u64)
    );
}

#[test]
fn a_ledger_envelope_is_opened_in_rfc_0003_order() {
    let ledger = VerifierLedger::decode_payload(
        &ledger_payload(&[record(1, &[0x01], &[5])]),
        &DecodeLimits::MESSAGE,
    )
    .unwrap();
    let tvl = ledger.seal().unwrap();
    let open = |bytes: &[u8]| {
        VerifierLedger::open(bytes, &DecodeLimits::MESSAGE).map_err(|e| (e.code(), e.offset()))
    };
    assert_eq!(open(&tvl), Ok(ledger.clone()));
    // The header: magic, envelope version 1, schema 0x0701, version 1, flags 0 (03 §5).
    assert_eq!(&tvl[..12], b"TSRA\x01\x00\x01\x07\x01\x00\x00\x00");
    assert_eq!(
        &sample_with(&[]).seal().unwrap()[..12],
        b"TSRA\x01\x00\x03\x04\x01\x00\x00\x00"
    );
    // A sample's schema, another version, a changed digest, a changed payload byte.
    let mut wrong = tvl.clone();
    wrong[6..8].copy_from_slice(&0x0403u16.to_le_bytes());
    assert_eq!(open(&wrong), Err((ErrorCode::Magic, 6)));
    let mut wrong = tvl.clone();
    wrong[8..10].copy_from_slice(&2u16.to_le_bytes());
    assert_eq!(open(&wrong), Err((ErrorCode::Version, 8)));
    let digest_at = tvl.len() - 32;
    for at in [digest_at, 20] {
        let mut wrong = tvl.clone();
        wrong[at] ^= 1;
        assert_eq!(
            open(&wrong),
            Err((ErrorCode::Digest, digest_at as u64)),
            "{at}"
        );
    }
    // A sample does not open as a ledger, nor a ledger as a sample.
    let tas = sample_with(&[key(1, 1)]).seal().unwrap();
    assert_eq!(open(&tas), Err((ErrorCode::Magic, 6)));
    assert_eq!(
        AuditSample::open(&tvl, &DecodeLimits::MESSAGE).map_err(|e| (e.code(), e.offset())),
        Err((ErrorCode::Magic, 6))
    );
}

#[test]
fn a_bit_set_has_one_encoding_per_set() {
    let mut set = BitSet::new();
    assert!(set.is_empty() && set.as_bytes().is_empty());
    for i in [0, 7, 8, 63, 64, 1000] {
        set.insert(i).unwrap();
        assert!(set.contains(i));
    }
    set.insert(7).unwrap();
    assert_eq!(set.len(), 6);
    assert_eq!(set.iter().collect::<Vec<_>>(), [0, 7, 8, 63, 64, 1000]);
    assert!(!set.contains(1) && !set.contains(1001) && !set.contains(u64::MAX));
    assert_eq!(set.as_bytes().len(), 126);
    assert_eq!(*set.as_bytes().last().unwrap(), 1 << 0);
}

// ----- The ledger's file -----------------------------------------------------------------------

#[test]
fn the_ledger_file_is_held_by_one_writer_and_written_atomically() {
    let dir = scratch("file");
    let path = dir.join("ledger.tvl");
    let (scenario, sample) = sample_of_five();
    let manifest = *scenario.period.manifest_digest();
    {
        let mut file = LedgerFile::create(&path).unwrap();
        assert_eq!(file.ledger(), &VerifierLedger::new());
        // A second writer is kept out while the first holds the file.
        assert_eq!(
            LedgerFile::open(&path, &DecodeLimits::MESSAGE).err(),
            Some(VerifierError { code: Code::Io })
        );
        file.record_accepted(
            &sample,
            &statement(manifest, vec![req(1, 3)]),
            &[FixedBits(9)],
        )
        .unwrap();
        // The replay is rejected and the file is unchanged.
        let bytes = fs::read(&path).unwrap();
        assert_eq!(
            file.record_accepted(
                &sample,
                &statement(manifest, vec![req(1, 3)]),
                &[FixedBits(9)]
            ),
            Err(Failure::Rejected(batch_mismatch()))
        );
        assert_eq!(fs::read(&path).unwrap(), bytes);
    }
    // Dropped, the file is free again, and holds what was recorded.
    let file = LedgerFile::open(&path, &DecodeLimits::MESSAGE).unwrap();
    let record = file.ledger().proven(sample.identity()).unwrap();
    assert_eq!(record.iter().collect::<Vec<_>>(), [(2, FixedBits(9))]);
    assert_eq!(
        &read_ledger(&path, &DecodeLimits::MESSAGE).unwrap(),
        file.ledger()
    );
    drop(file);
    // A ledger is never created over an existing file.
    assert_eq!(
        LedgerFile::create(&path).err(),
        Some(VerifierError { code: Code::Usage })
    );
    // A file that does not decode, one too large, one missing.
    fs::write(&path, b"TSRA").unwrap();
    assert_eq!(
        read_ledger(&path, &DecodeLimits::MESSAGE),
        Err(VerifierError {
            code: Code::CodecTruncated
        })
    );
    let small = DecodeLimits {
        max_total_bytes: 3,
        ..DecodeLimits::MESSAGE
    };
    assert_eq!(
        read_ledger(&path, &small),
        Err(VerifierError {
            code: Code::CodecLimit
        })
    );
    fs::remove_file(&path).unwrap();
    assert_eq!(
        read_ledger(&path, &DecodeLimits::MESSAGE),
        Err(VerifierError { code: Code::Io })
    );
    fs::remove_dir(&dir).unwrap();
}

// ----- The report ------------------------------------------------------------------------------

#[test]
fn a_missing_proof_is_unproven_at_the_deadline() {
    let (scenario, sample) = sample_of_five();
    let manifest = *scenario.period.manifest_digest();
    let mut ledger = VerifierLedger::new();
    // Nothing proven: every request is unproven, and nothing has a Û.
    let report = AuditReport::from_ledger(&sample, &ledger).unwrap();
    assert!(
        report
            .outcomes()
            .iter()
            .all(|(_, o)| *o == RequestOutcome::Unproven(UnprovenReason::DeadlineMissed))
    );
    assert_eq!(report.summary().unproven, 5);
    assert_eq!(report.summary().max_uhat, None);
    ledger
        .record_accepted(
            &sample,
            &statement(manifest, vec![req(1, 2), req(1, 4)]),
            &[FixedBits(3), FixedBits(11)],
        )
        .unwrap();
    let report = AuditReport::from_ledger(&sample, &ledger).unwrap();
    assert_eq!(report.sample(), &sample.seed());
    assert_eq!(report.manifest_digest(), &manifest);
    let outcomes: Vec<_> = report
        .outcomes()
        .iter()
        .map(|(k, o)| (*k, o.clone()))
        .collect();
    let missed = RequestOutcome::Unproven(UnprovenReason::DeadlineMissed);
    assert_eq!(
        outcomes,
        [
            (key(1, 1), missed.clone()),
            (key(1, 2), RequestOutcome::Accepted { uhat: FixedBits(3) }),
            (key(1, 3), missed.clone()),
            (
                key(1, 4),
                RequestOutcome::Accepted {
                    uhat: FixedBits(11)
                }
            ),
            (key(1, 5), missed),
        ]
    );
    let s = report.summary();
    assert_eq!(
        (
            s.selected,
            s.accepted,
            s.rejected,
            s.unproven,
            s.inconclusive,
            s.max_uhat
        ),
        (5, 2, 0, 3, 0, Some(FixedBits(11)))
    );
    let json = Json::parse(&report.to_json().unwrap()).unwrap();
    assert_eq!(
        json.get("schema").and_then(Json::as_str),
        Some("tessera.audit-report/1")
    );
    let first = &json.get("outcomes").and_then(Json::as_array).unwrap()[0];
    assert_eq!(
        first.get("outcome").and_then(Json::as_str),
        Some("unproven")
    );
    assert_eq!(
        first.get("unproven").and_then(Json::as_str),
        Some("DeadlineMissed")
    );
}

#[test]
fn a_ledger_record_past_the_sample_is_refused() {
    let sample = sample_with(&[key(1, 1), key(1, 2)]);
    // A record of this sample naming index 8.
    let mut proven = sample.identity().to_vec();
    proven.extend(2u64.to_le_bytes());
    proven.extend([0x00, 0x01]);
    proven.extend(1u64.to_le_bytes());
    proven.extend(5u64.to_le_bytes());
    let ledger =
        VerifierLedger::decode_payload(&ledger_payload(&[proven]), &DecodeLimits::MESSAGE).unwrap();
    assert_eq!(
        AuditReport::from_ledger(&sample, &ledger),
        Err(inconclusive(Code::Usage))
    );
    // Outcomes not one per selected request.
    let accepted = RequestOutcome::Accepted { uhat: FixedBits(0) };
    for n in [0, 1, 3] {
        assert_eq!(
            AuditReport::new(&sample, vec![accepted.clone(); n]),
            Err(inconclusive(Code::Usage))
        );
    }
}

#[test]
fn the_report_renders_each_outcome_as_the_example_does() {
    let example: Json = {
        let path = tv_testkit::vectors_dir().join("../../docs/perf/examples/audit-report.json");
        Json::parse(&fs::read_to_string(path).unwrap()).unwrap()
    };
    let keys: Vec<RequestKey> = example
        .get("outcomes")
        .and_then(Json::as_array)
        .unwrap()
        .iter()
        .map(|o| RequestKey {
            session_id: tv_testkit::hex(o.get("session_id").and_then(Json::as_str).unwrap())
                .try_into()
                .unwrap(),
            request_id: o.get("request_id").and_then(Json::as_u64).unwrap(),
        })
        .collect();
    let sample = sample_with(&keys);
    let report = AuditReport::new(
        &sample,
        vec![
            RequestOutcome::Accepted { uhat: FixedBits(0) },
            RequestOutcome::Rejected(RejectReason::Anchor {
                request_id: Some(7),
                kind: AnchorFailure::Frames,
            }),
            RequestOutcome::Unproven(UnprovenReason::DeadlineMissed),
            RequestOutcome::Inconclusive(VerifierError { code: Code::Memory }),
        ],
    )
    .unwrap();
    let json = Json::parse(&report.to_json().unwrap()).unwrap();
    assert_eq!(json.get("outcomes"), example.get("outcomes"));
    assert_eq!(json.get("summary"), example.get("summary"));
    assert_eq!(json.get("schema"), example.get("schema"));
    let hexed = |d: &[u8]| d.iter().map(|b| format!("{b:02x}")).collect::<String>();
    assert_eq!(
        json.get("sample").and_then(Json::as_str),
        Some(hexed(&sample.seed()).as_str())
    );
    assert_eq!(
        json.get("manifest").and_then(Json::as_str),
        Some(hexed(sample.manifest_digest()).as_str())
    );
    // The other unproven reasons' names.
    assert_eq!(
        UnprovenReason::ProverReportedMismatch.name(),
        "ProverReportedMismatch"
    );
    assert_eq!(
        UnprovenReason::ProverReportedAnchorMismatch.name(),
        "ProverReportedAnchorMismatch"
    );
}

// ----- Seeded cases ------------------------------------------------------------------------------

/// RFC-0004's `H`, from the test kit's BLAKE3, not `tv-hash`'s.
fn reference_h(domain: &str, input: &[u8]) -> [u8; 32] {
    blake3::derive_key(&format!("tessera/v1/{domain}"), input)
}

/// RFC-0016 step 4 read on its own: a linear scan for the bucket, the value from the reference
/// hash.
fn reference_selects(buckets: &[(u32, u64)], seed: &[u8; 32], k: &RequestKey, tokens: u64) -> bool {
    let value = {
        let mut input = seed.to_vec();
        input.extend(k.session_id);
        input.extend(k.request_id.to_le_bytes());
        u64::from_le_bytes(reference_h("audit-select", &input)[..8].try_into().unwrap())
    };
    match buckets.iter().find(|&&(max, _)| tokens <= u64::from(max)) {
        Some(&(_, threshold)) => value < threshold,
        None => true,
    }
}

#[test]
fn a_thousand_seeds_select_as_the_reference_does() {
    let mut rng = SplitMix64::new(0x0151_0016_0403_0701);
    let bytes = |n: usize, rng: &mut SplitMix64| -> Vec<u8> {
        (0..n.div_ceil(8))
            .flat_map(|_| rng.next_u64().to_le_bytes())
            .take(n)
            .collect()
    };
    let (mut chosen, mut total) = (0usize, 0usize);
    for seed_index in 0..1000 {
        // A policy of up to four buckets, a toss, a manifest and up to 24 candidates.
        let n_buckets = (rng.next_u64() % 5) as usize;
        let mut buckets = Vec::new();
        let mut max = 0u32;
        for _ in 0..n_buckets {
            max = max.saturating_add(1 + (rng.next_u64() % 50_000) as u32);
            buckets.push((max, rng.next_u64()));
        }
        let p = policy(&buckets);
        let t = toss(
            bytes(32, &mut rng).try_into().unwrap(),
            bytes(32, &mut rng).try_into().unwrap(),
        );
        let manifest: [u8; 32] = bytes(32, &mut rng).try_into().unwrap();
        let seed = t.sample_seed(&manifest);
        let reference_seed = reference_h(
            "audit-sample",
            &[&t.verifier_share[..], &t.prover_share, &manifest].concat(),
        );
        assert_eq!(seed, reference_seed, "seed {seed_index}");
        let candidates: Vec<Candidate> = (0..rng.next_u64() % 25)
            .map(|_| Candidate {
                session_id: bytes(16, &mut rng).try_into().unwrap(),
                request_id: rng.next_u64(),
                ended: true,
                tokens: match rng.next_u64() % 4 {
                    0 => u64::MAX,
                    1 => u64::from(max) + 1,
                    _ => rng.next_u64() % (u64::from(max) + 2),
                },
            })
            .collect();
        let mut expected: Vec<RequestKey> = candidates
            .iter()
            .map(|c| {
                (
                    RequestKey {
                        session_id: c.session_id,
                        request_id: c.request_id,
                    },
                    c.tokens,
                )
            })
            .filter(|(k, tokens)| reference_selects(&buckets, &seed, k, *tokens))
            .map(|(k, _)| k)
            .collect();
        expected.sort();
        assert_eq!(
            select(&p, &seed, &candidates).unwrap(),
            expected,
            "seed {seed_index}"
        );
        chosen += expected.len();
        total += candidates.len();
    }
    assert!(chosen > 0 && chosen < total, "{chosen} of {total}");
}

#[test]
fn each_buckets_selected_fraction_is_within_its_binomial_interval() {
    // 3 × 60,000 candidates at rates 1/2, 1/10 and 1/250, against 99.9% intervals (z = 3.29).
    let rates = [(1u32 << 10, 2u64), (1 << 14, 10), (1 << 17, 250)];
    let buckets: Vec<(u32, u64)> = rates.iter().map(|&(m, d)| (m, u64::MAX / d)).collect();
    let p = policy(&buckets);
    let seed = toss([8; 32], [9; 32]).sample_seed(&[1; 32]);
    let n = 60_000u64;
    for (b, &(max, d)) in rates.iter().enumerate() {
        let candidates: Vec<Candidate> = (0..n)
            .map(|i| Candidate {
                session_id: [b as u8; 16],
                request_id: i,
                ended: true,
                tokens: u64::from(max),
            })
            .collect();
        let k = select(&p, &seed, &candidates).unwrap().len() as f64;
        let rate = 1.0 / d as f64;
        let mean = n as f64 * rate;
        let spread = 3.29 * (n as f64 * rate * (1.0 - rate)).sqrt();
        assert!(
            (k - mean).abs() <= spread,
            "bucket {b}: {k} against {mean} ± {spread}"
        );
    }
}

#[test]
fn one_byte_of_either_share_changes_about_half_the_decisions() {
    let p = halves();
    let candidates: Vec<Candidate> = (0..20_000u64)
        .map(|i| Candidate {
            session_id: [3; 16],
            request_id: i,
            ended: true,
            tokens: 100,
        })
        .collect();
    let base = toss([1; 32], [2; 32]);
    let decisions = |t: &CoinTossTranscript| {
        let seed = t.sample_seed(&[5; 32]);
        candidates
            .iter()
            .map(|c| {
                p.selects(
                    selection_value(
                        &seed,
                        &RequestKey {
                            session_id: c.session_id,
                            request_id: c.request_id,
                        },
                    ),
                    c.tokens,
                )
            })
            .collect::<Vec<bool>>()
    };
    let original = decisions(&base);
    let mut prover = base;
    prover.prover_share[0] ^= 1;
    let mut verifier = toss([1; 32], [2; 32]);
    verifier.verifier_share[17] ^= 0x80;
    for changed in [prover, verifier] {
        let flipped = decisions(&changed)
            .iter()
            .zip(&original)
            .filter(|(a, b)| a != b)
            .count() as f64;
        // Each decision flips with probability 1/2: 10,000 ± 3.29 · 70.7.
        assert!((flipped - 10_000.0).abs() <= 233.0, "{flipped}");
    }
}

#[test]
fn seeded_samples_and_ledgers_round_trip_and_every_prefix_is_cut_short() {
    let mut rng = SplitMix64::new(0x7e55_e2a0_0151);
    for case in 0..40 {
        let mut keys: Vec<RequestKey> = (0..rng.next_u64() % 9)
            .map(|_| RequestKey {
                session_id: [(rng.next_u64() % 4) as u8; 16],
                request_id: rng.next_u64() % 1000,
            })
            .collect();
        keys.sort();
        keys.dedup();
        let sample = sample_with(&keys);
        let payload = sample.encode_payload().unwrap();
        assert_eq!(
            AuditSample::decode_payload(&payload, &DecodeLimits::MESSAGE).unwrap(),
            sample
        );
        // A ledger with a record for some of the sample's requests.
        let mut ledger = VerifierLedger::new();
        let some: Vec<RequestRef> = keys
            .iter()
            .filter(|_| rng.next_u64().is_multiple_of(2))
            .map(|k| RequestRef {
                request_id: k.request_id,
                session_id: k.session_id,
                prompt_len: 1,
                output_len: 1,
            })
            .collect();
        let uhat: Vec<FixedBits> = some.iter().map(|_| FixedBits(rng.next_u64())).collect();
        ledger
            .record_accepted(&sample, &statement([7; 32], some), &uhat)
            .unwrap();
        let bytes = ledger.encode_payload().unwrap();
        assert_eq!(
            VerifierLedger::decode_payload(&bytes, &DecodeLimits::MESSAGE).unwrap(),
            ledger,
            "case {case}"
        );
        for cut in 0..bytes.len() {
            assert_eq!(
                refusal(&bytes[..cut]).0,
                ErrorCode::Truncated,
                "case {case}: {cut}"
            );
        }
        let report = AuditReport::from_ledger(&sample, &ledger).unwrap();
        assert_eq!(report.summary().accepted, uhat.len() as u64);
        assert_eq!(report.summary().selected, keys.len() as u64);
    }
}

#[test]
fn the_vectors_samples_report_every_request_unproven_with_an_empty_ledger() {
    let doc = read_vectors("audit/audit.json");
    for case in doc.get("samples").and_then(Json::as_array).unwrap() {
        let envelope = tv_testkit::hex(case.get("envelope").and_then(Json::as_str).unwrap());
        let sample = AuditSample::open(&envelope, &DecodeLimits::MESSAGE).unwrap();
        let report = AuditReport::from_ledger(&sample, &VerifierLedger::new()).unwrap();
        assert_eq!(report.summary().unproven, sample.selected().len() as u64);
        assert_eq!(report.summary().accepted, 0);
    }
}

#[test]
fn a_request_listed_twice_for_batching_is_refused_and_bounds_hold() {
    use tv_core::audit::{BatchRequest, pack};
    let r = |s: u8, tokens: u64| BatchRequest {
        key: key(s, 1),
        group: 0u8,
        tokens,
    };
    assert_eq!(
        pack(10, &[r(1, 3), r(2, 3), r(1, 4)]).err(),
        Some(inconclusive(Code::Usage))
    );
    // A request longer than the maximum takes no other; ties go in key order; exactly the
    // maximum fits.
    let packed = pack(10, &[r(1, 11), r(2, 5), r(3, 5), r(4, 0)]).unwrap();
    let shapes: Vec<(Vec<RequestKey>, u64)> = packed
        .iter()
        .map(|b| (b.requests.clone(), b.tokens))
        .collect();
    assert_eq!(
        shapes,
        [
            (vec![key(1, 1)], 11),
            (vec![key(2, 1), key(3, 1), key(4, 1)], 10)
        ]
    );
}

#[test]
fn an_accepted_batch_of_no_request_changes_nothing() {
    let (scenario, sample) = sample_of_five();
    let mut ledger = VerifierLedger::new();
    let empty = statement(*scenario.period.manifest_digest(), Vec::new());
    assert_eq!(ledger.check_accounting(&sample, &empty), Ok(Vec::new()));
    ledger.record_accepted(&sample, &empty, &[]).unwrap();
    assert_eq!(ledger, VerifierLedger::new());
    assert!(ledger.proven(sample.identity()).is_none());
}

#[test]
fn a_large_batch_is_checked_in_quasi_linear_time() {
    // 2,000,000 selected requests, all in one batch, then the same with a repeat at both ends.
    // A pairwise search for repeats would make 2·10^12 comparisons, minutes even vectorized;
    // the sort takes well under a second, so the bound leaves a wide margin for a loaded host.
    // Only the checks are timed.
    let n = 2_000_000u64;
    let keys: Vec<RequestKey> = (0..n)
        .map(|r| RequestKey {
            session_id: [1; 16],
            request_id: r,
        })
        .collect();
    let sample = sample_with(&keys);
    let ledger = VerifierLedger::new();
    let mut requests: Vec<RequestRef> = keys.iter().map(|k| req(1, k.request_id)).collect();
    requests.reverse();
    let honest = statement([7; 32], requests.clone());
    requests.push(req(1, n - 1));
    let repeated = statement([7; 32], requests);
    let start = std::time::Instant::now();
    let indices = ledger.check_accounting(&sample, &honest).unwrap();
    let elapsed = start.elapsed();
    assert_eq!(indices.len() as u64, n);
    assert_eq!(indices[0], n - 1);
    assert_eq!(
        ledger.check_accounting(&sample, &repeated),
        Err(batch_rejected())
    );
    assert!(elapsed < std::time::Duration::from_secs(30), "{elapsed:?}");
}

#[test]
fn a_rate_has_at_most_nineteen_digits_whatever_its_whole_part() {
    // The vectors' 20-digit refusal has a whole part of 0, which the 64-bit denominator refuses
    // too; a whole part of 1 with 20 zeros is refused by the grammar alone.
    for (rate, threshold) in [
        ("1.0000000000000000000", Some(u64::MAX)),
        ("1.00000000000000000000", None),
        ("0.0000000000000000000", Some(0)),
        ("0.00000000000000000000", None),
        ("1.", None),
        ("1.01", None),
    ] {
        assert_eq!(RateBucket::threshold_of(rate), threshold, "{rate}");
    }
}

#[test]
fn a_selection_value_equal_to_its_threshold_is_not_selected() {
    let seed = toss([1; 32], [2; 32]).sample_seed(&[3; 32]);
    let k = key(4, 5);
    let value = selection_value(&seed, &k);
    let candidate = [Candidate {
        session_id: k.session_id,
        request_id: k.request_id,
        ended: true,
        tokens: 1,
    }];
    // Selected iff the value is below the threshold: not at it, just above it.
    assert!(
        select(&policy(&[(1, value)]), &seed, &candidate)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        select(&policy(&[(1, value + 1)]), &seed, &candidate).unwrap(),
        [k]
    );
    assert!(!policy(&[(1, value)]).selects(value, 1));
    assert!(policy(&[(1, value + 1)]).selects(value, 1));
}
