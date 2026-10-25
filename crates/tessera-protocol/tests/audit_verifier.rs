// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The prover's audit sampling against the independent verifier's (RFC-0016 'Sample
//! derivation', INV-AUD-2, #151): over 1,000 seeds, with random policies, coin tosses, manifests
//! and candidates, both derive the same sample seed, the same selection value for every
//! candidate and the same selected requests. The verifier decodes the prover's sample and
//! encodes it back to the same bytes. A differential test of the prover's code, so it lives on
//! the prover's side (DEP-2).

use tessera_codec::{Ascii, encode};
use tessera_hash::ParamSetId;
use tessera_protocol::{
    AuditPolicy, AuditSample, CoinTossTranscript, RateBucket, RequestKey, coin_commitment,
    sample_seed, selection_value,
};
use tv_codec::DecodeLimits;
use tv_core::anchor::Candidate;
use tv_core::audit as tv;
use tv_core::claims::ParamSetId as TvParamSetId;

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

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn bytes<const N: usize>(&mut self) -> [u8; N] {
        let mut out = [0u8; N];
        for chunk in out.chunks_mut(8) {
            let word = self.next().to_le_bytes();
            chunk.copy_from_slice(&word[..chunk.len()]);
        }
        out
    }
}

/// One to four buckets of increasing length, with thresholds from 0 to `2^64 − 1`, an edge
/// among them now and then.
fn buckets(rng: &mut Rng) -> Vec<(u32, u64)> {
    let mut max_tokens = 0u32;
    (0..=rng.below(4))
        .map(|_| {
            max_tokens += 1 + u32::try_from(rng.below(5_000)).expect("small");
            let threshold = match rng.below(8) {
                0 => 0,
                1 => u64::MAX,
                _ => rng.next(),
            };
            (max_tokens, threshold)
        })
        .collect()
}

#[test]
fn prover_and_verifier_select_the_same_sample_for_1000_seeds() {
    let mut rng = Rng(151);
    let limits = DecodeLimits::MESSAGE;
    let mut selected_total = 0usize;
    let mut shared_sessions = 0usize;
    for _ in 0..1_000 {
        let buckets = buckets(&mut rng);
        let last = buckets.last().map_or(0, |b| b.0);
        let max_batch = 2 + u32::try_from(rng.below(10_000)).expect("small");
        let min_batch = u32::try_from(rng.below(u64::from(max_batch / 2) + 1)).expect("small");
        let (period, deadline) = (rng.next(), rng.next());
        let policy = AuditPolicy::new(
            period,
            buckets
                .iter()
                .map(|&(max_tokens, threshold)| RateBucket {
                    max_tokens,
                    threshold,
                })
                .collect(),
            min_batch,
            max_batch,
            deadline,
            ParamSetId(Ascii::new("live-64/1").expect("ASCII")),
        )
        .expect("a valid policy");
        let tv_policy = tv::AuditPolicy::new(
            period,
            buckets
                .iter()
                .map(|&(max_tokens, threshold)| tv::RateBucket {
                    max_tokens,
                    threshold,
                })
                .collect(),
            min_batch,
            max_batch,
            deadline,
            TvParamSetId::new("live-64/1").expect("a name"),
        )
        .expect("a valid policy");

        let verifier_share: [u8; 32] = rng.bytes();
        let toss = CoinTossTranscript {
            verifier_commitment: coin_commitment(&verifier_share),
            prover_share: rng.bytes(),
            verifier_share,
        };
        let tv_toss = tv::CoinTossTranscript {
            verifier_commitment: toss.verifier_commitment,
            prover_share: toss.prover_share,
            verifier_share: toss.verifier_share,
        };
        let manifest: [u8; 32] = rng.bytes();
        let seed = sample_seed(&toss, &manifest);
        assert_eq!(tv_toss.sample_seed(&manifest), seed);

        // Candidates up to past every bucket, some keys twice with two lengths. Keys share a few
        // sessions, with ids around multiples of 256 whose numeric and little-endian byte orders
        // differ (RFC-0067 'Request order').
        let sessions: Vec<[u8; 16]> = (0..1 + rng.below(3)).map(|_| rng.bytes()).collect();
        let mut candidates: Vec<(RequestKey, u64)> = Vec::new();
        for _ in 0..rng.below(48) {
            let session = sessions[rng.below(sessions.len() as u64) as usize];
            let request_id = match rng.below(3) {
                0 => rng.next(),
                _ => 256 * rng.below(4) + rng.below(3),
            };
            let key = RequestKey {
                session_id: session,
                request_id,
            };
            let tokens = rng.below(u64::from(last) + 64);
            candidates.push((key, tokens));
            if rng.below(8) == 0 {
                candidates.push((key, rng.below(u64::from(last) + 64)));
            }
        }
        let tv_candidates: Vec<Candidate> = candidates
            .iter()
            .map(|&(key, tokens)| Candidate {
                session_id: key.session_id,
                request_id: key.request_id,
                ended: true,
                tokens,
            })
            .collect();
        for (key, _) in &candidates {
            let tv_key = tv::RequestKey {
                session_id: key.session_id,
                request_id: key.request_id,
            };
            assert_eq!(
                tv::selection_value(&seed, &tv_key),
                selection_value(&seed, key)
            );
        }

        let sample = AuditSample::draw(manifest, policy, toss, candidates).expect("it opens");
        let tv_selected = tv::select(&tv_policy, &seed, &tv_candidates).expect("memory");
        let keys = |keys: &[tv::RequestKey]| -> Vec<RequestKey> {
            keys.iter()
                .map(|k| RequestKey {
                    session_id: k.session_id,
                    request_id: k.request_id,
                })
                .collect()
        };
        assert_eq!(keys(&tv_selected), sample.selected());
        selected_total += sample.selected().len();
        shared_sessions += sample
            .selected()
            .windows(2)
            .filter(|w| w[0].session_id == w[1].session_id)
            .count();

        // The verifier reads the prover's sample, and writes the same bytes.
        let bytes = encode(&sample);
        let decoded = tv::AuditSample::decode_payload(&bytes, &limits).expect("the sample");
        assert_eq!(keys(decoded.selected()), sample.selected());
        assert_eq!(decoded.seed(), seed);
        assert_eq!(decoded.encode_payload().expect("memory"), bytes);
    }
    // The comparison is not of empty samples, and samples hold keys of one session.
    assert!(selected_total > 1_000, "{selected_total}");
    assert!(shared_sessions > 100, "{shared_sessions}");
}
