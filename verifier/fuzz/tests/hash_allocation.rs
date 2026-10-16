// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The memory `tv-hash` allocates, counted by the `cap` allocator: checking a multiproof or an
//! inclusion proof, hashing, chaining a column and replaying a transcript allocate nothing,
//! whatever the proof holds, so no input can make them allocate (RFC-0019 'Robustness'). The
//! only allocations are the ones a caller asks for by size: `challenge_indices` reserves exactly
//! its indices, and `SessionContext::to_tce` its encoding. Memory too short for the indices is
//! `TranscriptError::Memory`, before any squeeze.
//!
//! The allocator's counters and limit are global, so the cases run one after another in a single
//! test.

use std::alloc::System;
use std::num::NonZeroU32;

use cap::Cap;
use tv_field::Fp;
use tv_fuzz::proofs::{MAX_LOG_N, canonical_multiproof, inclusion_case, multiproof_case, trees};
use tv_hash::column::column_digest;
use tv_hash::merkle::verify_multiproof;
use tv_hash::rfc6962::verify_inclusion;
use tv_hash::transcript::{Ascii, Mode, ParamSetId, SessionContext, Transcript, TranscriptError};
use tv_hash::{Domain, DomainHasher, hash};

#[global_allocator]
static ALLOCATOR: Cap<System> = Cap::new(System, usize::MAX);

/// The bytes `f` allocates in all, and its result.
fn allocated_by<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let before = ALLOCATOR.total_allocated();
    let result = f();
    (result, ALLOCATOR.total_allocated() - before)
}

#[test]
fn proof_checks_hashing_and_transcripts_allocate_nothing() {
    // Proofs of every shape the fuzz scripts build: honest, and changed in each way.
    for seed in 0u16..400 {
        let script: Vec<u8> = (0..64u16)
            .map(|i| (seed.wrapping_mul(31).wrapping_add(i.wrapping_mul(seed | 1)) % 251) as u8)
            .collect();
        let case = multiproof_case(&script);
        let (_, bytes) =
            allocated_by(|| verify_multiproof(&case.root, case.n, &case.leaves, &case.siblings));
        assert_eq!(bytes, 0, "verify_multiproof, script {seed}");
        let case = inclusion_case(&script);
        let (_, bytes) = allocated_by(|| {
            verify_inclusion(case.index, case.size, &case.leaf, &case.proof, &case.root)
        });
        assert_eq!(bytes, 0, "verify_inclusion, script {seed}");
    }
    // The largest openings of the largest tree: every leaf, and every other leaf.
    let tree = &trees()[usize::from(MAX_LOG_N)];
    for step in [1, 2] {
        let indices: Vec<u64> = (0..tree.n()).step_by(step).collect();
        let leaves: Vec<_> = indices
            .iter()
            .map(|&i| (i, tree.levels[0][i as usize]))
            .collect();
        let siblings = canonical_multiproof(&tree.levels, &indices);
        let (outcome, bytes) =
            allocated_by(|| verify_multiproof(&tree.root(), tree.n(), &leaves, &siblings));
        assert_eq!((outcome, bytes), (Ok(()), 0), "every {step} leaves");
    }

    let input = vec![0x5a; (1 << 20) + 77];
    let (_, bytes) = allocated_by(|| hash(Domain::Envelope, &input));
    assert_eq!(bytes, 0, "hash of 1 MiB");
    let (_, bytes) = allocated_by(|| {
        let mut hasher = DomainHasher::new(Domain::Table);
        for piece in input.chunks(1000) {
            hasher.update(piece);
        }
        hasher.finalize()
    });
    assert_eq!(bytes, 0, "DomainHasher");
    let symbols: Vec<Fp> = (0..5000).map(Fp::reduce).collect();
    for block in [1, 32, 124, 125, 4096] {
        let (_, bytes) =
            allocated_by(|| column_digest(&[3; 32], 9, NonZeroU32::new(block).unwrap(), &symbols));
        assert_eq!(bytes, 0, "column_digest, B = {block}");
    }

    let context = SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("ni-100/1").unwrap()),
        statement_digest: [1; 32],
        session_id: [2; 16],
    };
    let label = Ascii::new("pcs/advice/queries").unwrap();
    let (_, bytes) = allocated_by(|| {
        let mut transcript = Transcript::new(&context);
        transcript.absorb(label, &input);
        transcript.squeeze(label);
        transcript.challenge_fp(label).unwrap();
        transcript.challenge_fp2(label).unwrap();
        let _ = transcript
            .check_grind(label, 0, 7)
            .unwrap()
            .check_grind(label, 200, 7);
    });
    assert_eq!(bytes, 0, "transcript replay");
    let mut transcript = Transcript::new(&context);
    let (indices, bytes) = allocated_by(|| transcript.challenge_indices(label, 22, 193).unwrap());
    assert_eq!(
        (indices.len(), bytes),
        (193, 193 * 8),
        "exactly the indices"
    );
    // 2^32 − 1 indices, 32 GiB, with 1 MiB to spare: `Memory`, and no squeeze.
    let mut transcript = Transcript::new(&context);
    let state = *transcript.state();
    ALLOCATOR
        .set_limit(ALLOCATOR.allocated() + (1 << 20))
        .expect("a limit above the memory in use");
    let outcome = transcript.challenge_indices(label, 22, u32::MAX);
    ALLOCATOR.set_limit(usize::MAX).expect("no limit");
    assert_eq!(outcome, Err(TranscriptError::Memory));
    assert_eq!((*transcript.state(), transcript.counter()), (state, 0));
    let (tce, bytes) = allocated_by(|| context.to_tce());
    assert!(
        bytes >= tce.len() && bytes <= 2 * tce.len().next_power_of_two(),
        "{bytes}"
    );
}
