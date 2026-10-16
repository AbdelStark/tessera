// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The transcript's checks, each met exactly at its boundary (RFC-0004 'Transcript', RFC-0014
//! 'Grinding'): a nonce whose proof-of-work hash has exactly `k` leading zero bits passes for
//! `k` bits and fails for `k + 1`; a failed check absorbs nothing and gives the transcript up;
//! challenge indices take exactly `log_n` bits, and `log_n` above 64 changes nothing.

use tv_hash::transcript::{Ascii, Mode, ParamSetId, SessionContext, Transcript, TranscriptError};
use tv_hash::{Domain, DomainHasher, hash};

fn context() -> SessionContext<'static> {
    SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("ni-100/1").unwrap()),
        statement_digest: [0x33; 32],
        session_id: [0x44; 16],
    }
}

/// `H("pow", state ‖ Ascii(label) ‖ u64(nonce))`, written out here from RFC-0004.
fn pow_hash(state: &[u8; 32], label: &str, nonce: u64) -> [u8; 32] {
    let mut hasher = DomainHasher::new(Domain::Pow);
    hasher
        .update(state)
        .update(&(label.len() as u64).to_le_bytes())
        .update(label.as_bytes())
        .update(&nonce.to_le_bytes());
    hasher.finalize()
}

fn leading_zeros(digest: &[u8; 32]) -> u32 {
    let mut zeros = 0;
    for byte in digest {
        zeros += byte.leading_zeros();
        if *byte != 0 {
            break;
        }
    }
    zeros
}

#[test]
fn a_nonce_passes_exactly_the_bits_its_hash_has() {
    let label = Ascii::new("query-pow").unwrap();
    let state = *Transcript::new(&context()).state();
    let mut found = [None; 13];
    let mut nonce = 0u64;
    while found.iter().any(Option::is_none) {
        let zeros = leading_zeros(&pow_hash(&state, label.as_str(), nonce)) as usize;
        if zeros < found.len() && found[zeros].is_none() {
            found[zeros] = Some(nonce);
        }
        nonce += 1;
    }
    for (zeros, nonce) in found.iter().enumerate() {
        let nonce = nonce.unwrap();
        let zeros = zeros as u8;
        // Exactly `zeros` bits: accepted for every requirement up to `zeros`, and absorbed.
        for bits in 0..=zeros {
            let passed = Transcript::new(&context())
                .check_grind(label, bits, nonce)
                .unwrap_or_else(|e| panic!("nonce {nonce}, {bits} bits: {e}"));
            let mut absorbed = Transcript::new(&context());
            absorbed.absorb(label, &nonce.to_le_bytes());
            assert_eq!(
                passed, absorbed,
                "a valid nonce is absorbed under its label"
            );
        }
        // One bit more is a failure that reports both counts.
        for bits in [zeros + 1, zeros + 2, 255] {
            let failed = Transcript::new(&context())
                .check_grind(label, bits, nonce)
                .unwrap_err();
            assert_eq!(
                (failed.required(), failed.found()),
                (bits, u32::from(zeros))
            );
        }
    }
}

#[test]
fn the_proof_of_work_binds_the_state_the_label_and_the_nonce() {
    let label = Ascii::new("seed-pow").unwrap();
    // Find a nonce valid for 8 bits under one label, after one absorb.
    let mut transcript = Transcript::new(&context());
    transcript.absorb(Ascii::new("round1").unwrap(), b"commitments");
    let state = *transcript.state();
    let nonce = (0..)
        .find(|&n| leading_zeros(&pow_hash(&state, "seed-pow", n)) >= 8)
        .unwrap();
    let replay = || {
        let mut t = Transcript::new(&context());
        t.absorb(Ascii::new("round1").unwrap(), b"commitments");
        t
    };
    assert!(replay().check_grind(label, 8, nonce).is_ok());
    // The same nonce under another label, or from another state, is another hash: it fails
    // unless that hash happens to have 8 leading zero bits too, which the oracle decides.
    let other = Ascii::new("query-pow").unwrap();
    assert_eq!(
        replay().check_grind(other, 8, nonce).is_ok(),
        leading_zeros(&pow_hash(&state, "query-pow", nonce)) >= 8
    );
    let fresh = *Transcript::new(&context()).state();
    assert_eq!(
        Transcript::new(&context())
            .check_grind(label, 8, nonce)
            .is_ok(),
        leading_zeros(&pow_hash(&fresh, "seed-pow", nonce)) >= 8
    );
}

#[test]
fn indices_take_exactly_log_n_bits_and_a_larger_log_n_changes_nothing() {
    let label = Ascii::new("pcs/advice/queries").unwrap();
    for log_n in [65, 66, 128, u32::MAX] {
        let mut transcript = Transcript::new(&context());
        transcript.absorb(label, b"x");
        let before = (*transcript.state(), transcript.counter());
        assert_eq!(
            transcript.challenge_indices(label, log_n, 9),
            Err(TranscriptError::LogN { log_n })
        );
        assert_eq!((*transcript.state(), transcript.counter()), before);
    }
    // Every log_n from 0 to 64 masks the same squeezed words.
    let words = {
        let mut transcript = Transcript::new(&context());
        transcript.challenge_indices(label, 64, 8).unwrap()
    };
    for log_n in 0..=64u32 {
        let mut transcript = Transcript::new(&context());
        let indices = transcript.challenge_indices(label, log_n, 8).unwrap();
        let mask = if log_n == 64 {
            u64::MAX
        } else {
            (1 << log_n) - 1
        };
        let expected: Vec<u64> = words.iter().map(|w| w & mask).collect();
        assert_eq!(indices, expected, "log_n = {log_n}");
        assert_eq!(transcript.counter(), 2, "four words per squeeze");
    }
    // The words are the squeezes' little-endian words, in order.
    let mut transcript = Transcript::new(&context());
    let first = transcript.squeeze(label);
    let second = transcript.squeeze(label);
    let raw: Vec<u64> = [first, second]
        .iter()
        .flat_map(|out| {
            out.chunks(8)
                .map(|w| u64::from_le_bytes(w.try_into().unwrap()))
        })
        .collect();
    assert_eq!(raw, words);
}

#[test]
fn squeezes_follow_the_counter_and_the_ratchet() {
    let label = Ascii::new("tree/0/r/0").unwrap();
    let mut transcript = Transcript::new(&context());
    let state = *transcript.state();
    let out = transcript.squeeze(label);
    // out = H("transcript-squeeze", state ‖ Ascii(label) ‖ u64(0)).
    let mut input = state.to_vec();
    input.extend_from_slice(&(label.as_str().len() as u64).to_le_bytes());
    input.extend_from_slice(label.as_str().as_bytes());
    input.extend_from_slice(&0u64.to_le_bytes());
    assert_eq!(out, hash(Domain::TranscriptSqueeze, &input));
    // state = H("transcript-ratchet", state ‖ out).
    assert_eq!(
        *transcript.state(),
        hash(Domain::TranscriptRatchet, &[state, out].concat())
    );
    // The next squeeze under the same label differs: the counter and the state moved.
    assert_ne!(transcript.squeeze(label), out);
    assert_eq!(transcript.counter(), 2);
}
