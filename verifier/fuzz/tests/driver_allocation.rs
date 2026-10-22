// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The driver's input guard and public JSON writer fail without infallible allocation.
use cap::Cap;
use std::alloc::System;
use tv_codec::{DecodeLimits, WeightCommitmentPublic};
use tv_core::claims::ClaimList;
use tv_core::protocol::{FixedBits, NiProof};
use tv_core::{ErrorCode, RejectReason, Verdict, VerifierConfig, VerifierError};

#[global_allocator]
static ALLOCATOR: Cap<System> = Cap::new(System, usize::MAX);

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../crates/tv-core/tests/fixtures/driver")
            .join(name),
    )
    .unwrap()
}

#[test]
fn predecode_policy_decode_failures_and_report_serialization_respect_allocation_failures() {
    let limits = DecodeLimits::MESSAGE;
    let bytes = fixture("ni.tnp");
    let claims = ClaimList::open(&fixture("claims.tcl"), &limits).unwrap();
    let proof = NiProof::open(&bytes, &limits).unwrap();
    let weights = tv_codec::decode(
        &fixture("weights.tce"),
        &limits,
        WeightCommitmentPublic::decode,
    )
    .unwrap();
    let mut cfg = VerifierConfig {
        params: tv_core::params::ParamSet::Ni100,
        limits: Default::default(),
        accept_bound: FixedBits(0),
        weights,
    };
    let memory = Verdict::Inconclusive(VerifierError {
        code: ErrorCode::Memory,
    });
    cfg.limits.max_memory_bytes = bytes.len() as u64;
    ALLOCATOR.set_limit(ALLOCATOR.allocated()).unwrap();
    let guarded =
        tv_core::verify_non_interactive_bytes(&claims, &proof.round1.statement, &bytes, &cfg);
    ALLOCATOR.set_limit(usize::MAX).unwrap();
    assert_eq!(guarded.to_json().unwrap(), memory.to_json().unwrap());

    // Sweep the decoder's reservations, not the still-unhardened full arithmetic driver (#182).
    let mut spare = 0;
    let bound = bytes.len() * 8;
    let mut refused = 0;
    loop {
        ALLOCATOR.set_limit(ALLOCATOR.allocated() + spare).unwrap();
        let result = NiProof::open(&bytes, &limits);
        ALLOCATOR.set_limit(usize::MAX).unwrap();
        if let Err(error) = result {
            assert_eq!(error.code(), tv_codec::ErrorCode::Memory, "spare={spare}");
            refused += 1;
        } else if spare == bound {
            break;
        }
        assert!(
            spare < bound,
            "decoded fixture must fit the planning allowance"
        );
        spare = if spare == 0 {
            1
        } else {
            (spare * 2).min(bound)
        };
    }
    assert!(refused > 0);

    let rejected = Verdict::Reject(RejectReason::Params {
        name: claims.params.clone(),
    });
    ALLOCATOR.set_limit(ALLOCATOR.allocated()).unwrap();
    let json = rejected.to_json();
    ALLOCATOR.set_limit(usize::MAX).unwrap();
    assert_eq!(
        json,
        Err(VerifierError {
            code: ErrorCode::Memory
        })
    );
    assert!(rejected.to_json().is_ok());
}
