// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The `audit` target's oracle on seeded inputs, so that `cargo test` runs it without the
//! fuzzer: the samples of `spec/vectors/audit/`, their envelopes and the refused payloads, as
//! they are and with bytes changed, cut or added, ledgers built from the samples, and pending
//! tosses over the samples' manifests and policies, whose shares open their commitments under
//! the test kit's BLAKE3, payload and envelope.

use tv_core::audit::{AuditSample, VerifierLedger, VerifierToss};
use tv_core::protocol::{BatchDescriptor, FixedBits, PublicStatement, RequestRef};
use tv_fuzz::audit::check;
use tv_testkit::blake3::derive_key;
use tv_testkit::{Json, SplitMix64, hex, read_vectors};

/// RFC-0003's envelope of schema `0x0702` version 1 around a toss's `payload`, with the test
/// kit's BLAKE3.
fn seal_toss(payload: &[u8]) -> Vec<u8> {
    let mut out = [
        &b"TSRA"[..],
        &1u16.to_le_bytes(),
        &0x0702u16.to_le_bytes(),
        &1u16.to_le_bytes(),
        &0u16.to_le_bytes(),
        &(payload.len() as u64).to_le_bytes(),
        payload,
    ]
    .concat();
    let digest = derive_key("tessera/v1/envelope", &out);
    out.extend_from_slice(&digest);
    out
}

fn inputs() -> Vec<(u8, Vec<u8>)> {
    let doc = read_vectors("audit/audit.json");
    let mut out = Vec::new();
    for case in doc.get("samples").and_then(Json::as_array).unwrap() {
        let payload = hex(case.get("payload").and_then(Json::as_str).unwrap());
        let envelope = hex(case.get("envelope").and_then(Json::as_str).unwrap());
        let sample = AuditSample::open(&envelope, &tv_codec::DecodeLimits::MESSAGE).unwrap();
        // A ledger that derived the sample and accepted its first request, when it has one, from
        // a proof whose claim list names the policy's parameter set (RFC-0016 'The parameter
        // set').
        let mut deriving = vec![0; 24];
        deriving.extend(1u64.to_le_bytes());
        deriving.extend(sample.manifest_digest());
        deriving.extend(sample.identity());
        deriving.extend(0u64.to_le_bytes());
        let mut ledger =
            VerifierLedger::decode_payload(&deriving, &tv_codec::DecodeLimits::MESSAGE).unwrap();
        if let Some(first) = sample.selected().first() {
            let statement = PublicStatement {
                claim_list_digest: [0; 32],
                weights_root: [0; 32],
                weights_epoch: 0,
                manifest_digest: *sample.manifest_digest(),
                batch: BatchDescriptor {
                    requests: vec![RequestRef {
                        request_id: first.request_id,
                        session_id: first.session_id,
                        prompt_len: 1,
                        output_len: 1,
                    }],
                },
                predictor: tv_core::protocol::PredictorSpec::Exact {
                    decoding: tv_core::protocol::Decoding::Greedy,
                },
            };
            ledger
                .record_accepted(
                    &sample,
                    sample.policy().params(),
                    &statement,
                    &[FixedBits(3)],
                )
                .unwrap();
        }
        // A pending toss over the sample's manifest and policy (RFC-0067 'The pending toss'):
        // the payload's manifest and policy, the bytes before the coin toss and `selected`.
        let policy_end = payload.len() - 96 - 8 - 24 * sample.selected().len();
        let share = [payload.len() as u8; 32];
        let mut toss = payload[..policy_end].to_vec();
        toss.extend(derive_key("tessera/v1/coin-commit", &share));
        toss.extend(share);
        // Both readers accept it, so that the seeds reach the accepted path.
        let limits = tv_codec::DecodeLimits::MESSAGE;
        let opened = VerifierToss::open(&seal_toss(&toss), &limits).unwrap();
        assert_eq!(opened.manifest_digest(), sample.manifest_digest());
        assert_eq!(opened.policy(), sample.policy());
        VerifierToss::decode_payload(&toss, &limits).unwrap();
        out.push((4, toss.clone()));
        out.push((5, seal_toss(&toss)));
        out.push((0, payload));
        out.push((2, envelope));
        out.push((1, ledger.encode_payload().unwrap()));
        out.push((3, ledger.seal().unwrap()));
    }
    for case in doc.get("refused").and_then(Json::as_array).unwrap() {
        out.push((0, hex(case.get("payload").and_then(Json::as_str).unwrap())));
    }
    out
}

#[test]
fn seeded_inputs_meet_the_oracle() {
    let mut rng = SplitMix64::new(0xa0d1_7151_0701);
    let seeds = inputs();
    for (selector, input) in &seeds {
        for config in [0u8, 1] {
            let mut data = vec![*selector, config];
            data.extend(input);
            check(&data);
        }
    }
    for _ in 0..3000 {
        let (selector, base) = &seeds[(rng.next_u64() % seeds.len() as u64) as usize];
        let mut input = base.clone();
        match rng.next_u64() % 4 {
            0 if !input.is_empty() => {
                let at = (rng.next_u64() % input.len() as u64) as usize;
                input[at] ^= 1 << (rng.next_u64() % 8);
            }
            1 => input.truncate((rng.next_u64() % (input.len() as u64 + 1)) as usize),
            2 => input.push(rng.next_u64() as u8),
            _ => {
                let at = (rng.next_u64() % (input.len() as u64 + 1)) as usize;
                input.insert(at, rng.next_u64() as u8);
            }
        }
        let mut data = vec![*selector, (rng.next_u64() % 2) as u8];
        data.extend(input);
        check(&data);
    }
}
