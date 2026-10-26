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

#[cfg(unix)]
fn live_share_limits_precede_allocation_and_allocator_failures_are_fallible() {
    use tv_core::claims::ParamSetId;
    use tv_core::{LiveVerifier, params::ParamSet};

    let limits = DecodeLimits::MESSAGE;
    let mut claims = ClaimList::open(&fixture("claims.tcl"), &limits).unwrap();
    claims.params = ParamSetId::new("live-64/1").unwrap();
    claims.layout = ParamSet::Live64.layout();
    let proof = NiProof::open(&fixture("ni.tnp"), &limits).unwrap();
    let weights = tv_codec::decode(
        &fixture("weights.tce"),
        &limits,
        WeightCommitmentPublic::decode,
    )
    .unwrap();
    for (budget, spare, succeeds) in [(31, 64, false), (32, 31, false), (32, 32, true)] {
        let mut cfg = VerifierConfig {
            params: ParamSet::Live64,
            limits: Default::default(),
            accept_bound: FixedBits(0),
            weights,
        };
        cfg.limits.max_memory_bytes = budget;
        let mut live = LiveVerifier::new(claims.clone(), proof.round1.statement.clone(), cfg);
        let before = ALLOCATOR.total_allocated();
        ALLOCATOR.set_limit(ALLOCATOR.allocated() + spare).unwrap();
        let result = live.start();
        ALLOCATOR.set_limit(usize::MAX).unwrap();
        let allocated = ALLOCATOR.total_allocated() - before;
        if succeeds {
            assert!(result.is_ok());
            assert_eq!(allocated, 32);
        } else {
            assert!(matches!(
                result,
                Err(Verdict::Inconclusive(VerifierError {
                    code: ErrorCode::Memory
                }))
            ));
            assert_eq!(allocated, 0);
        }
    }
}

/// Runs on the process's only thread (`harness = false`): libtest would keep a thread of its own,
/// whose allocations the process-wide cap counts against this test's budget (#501).
fn main() {
    predecode_policy_decode_failures_and_report_serialization_respect_allocation_failures();
    #[cfg(unix)]
    live_share_limits_precede_allocation_and_allocator_failures_are_fallible();
    println!(
        "predecode_policy_decode_failures_and_report_serialization_respect_allocation_failures: ok"
    );
    #[cfg(unix)]
    println!(
        "live share: policy refusal and allocator failure allocate 0 bytes; success allocates 32"
    );
}
