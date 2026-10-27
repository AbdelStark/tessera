// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Allocation failures through the complete recorded test-small proof (RFC-0019, SP-8).
//! Each capped call has its own process: an infallible allocation must fail this test without
//! aborting its controller or hiding later diagnostics. This proves no production-size bound.

use cap::Cap;
use std::alloc::System;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use tv_codec::{DecodeLimits, WeightCommitmentPublic};
use tv_core::claims::ClaimList;
use tv_core::protocol::{FixedBits, NiProof};
use tv_core::{ErrorCode, Verdict, VerifierConfig};

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

fn worker(policy: bool, budget: usize) {
    let limits = DecodeLimits::MESSAGE;
    let bytes = fixture("ni.tnp");
    let claims = ClaimList::open(&fixture("claims.tcl"), &limits).unwrap();
    let statement = NiProof::open(&bytes, &limits).unwrap().round1.statement;
    let weights = tv_codec::decode(
        &fixture("weights.tce"),
        &limits,
        WeightCommitmentPublic::decode,
    )
    .unwrap();
    let mut cfg = VerifierConfig {
        params: tv_core::params::ParamSet::TestSmall,
        limits: Default::default(),
        accept_bound: FixedBits(0),
        weights,
    };
    if policy {
        cfg.limits.max_memory_bytes = budget as u64;
    }
    // Raise the live baseline to the historical peak so setup allocations cannot inflate the
    // measured verification peak. Inputs and this padding stay outside the work budget.
    let padding = vec![0u8; ALLOCATOR.max_allocated() - ALLOCATOR.allocated()];
    let baseline = ALLOCATOR.allocated();
    assert_eq!(baseline, ALLOCATOR.max_allocated());
    let total_before = ALLOCATOR.total_allocated();
    let allocator_budget = if policy { 64 << 20 } else { budget };
    ALLOCATOR
        .set_limit(baseline.checked_add(allocator_budget).unwrap())
        .unwrap();
    let verdict = tv_core::verify_non_interactive_bytes(&claims, &statement, &bytes, &cfg);
    let peak = ALLOCATOR.max_allocated() - baseline;
    let total = ALLOCATOR.total_allocated() - total_before;
    ALLOCATOR.set_limit(usize::MAX).unwrap();
    std::hint::black_box(&padding);
    assert!(peak <= budget);
    let outcome = match verdict {
        Verdict::Accept(_) => "accept",
        Verdict::Inconclusive(error) if error.code == ErrorCode::Memory => "memory",
        other => panic!("unexpected verdict: {other:?}"),
    };
    println!("{outcome} {peak} {total}");
}

fn run(policy: bool, budget: usize) -> (bool, usize) {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--worker", if policy { "policy" } else { "allocator" }])
        .arg(budget.to_string())
        .env("RUST_BACKTRACE", "0")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let started = Instant::now();
    while child.try_wait().unwrap().is_none() {
        if started.elapsed() > Duration::from_secs(30) {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("worker exceeded 30 seconds: policy={policy}, budget={budget}");
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "policy={policy}, budget={budget}: {}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let text = std::str::from_utf8(&output.stdout).unwrap();
    let fields = text.split_whitespace().collect::<Vec<_>>();
    assert_eq!(fields.len(), 3, "{text:?}");
    assert!(matches!(fields[0], "accept" | "memory"));
    (fields[0] == "accept", fields[1].parse().unwrap())
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    if let [_, worker_flag, mode, budget] = args.as_slice() {
        assert_eq!(worker_flag, "--worker");
        assert!(matches!(mode.as_str(), "policy" | "allocator"));
        worker(mode == "policy", budget.parse().unwrap());
        return;
    }
    assert_eq!(args.len(), 1);
    let (accepted, peak) = run(false, 64 << 20);
    assert!(
        accepted,
        "the recorded proof must pass without an allocation shortage"
    );
    println!("recorded proof work peak: {peak} bytes");
    let mut budgets = vec![0, 1, 31, 32];
    let upper = 64 << 20;
    let mut budget = 64;
    while budget <= upper {
        budgets.push(budget);
        budget *= 2;
    }
    budgets.extend([peak.saturating_sub(1), peak, peak + 1]);
    let mut allocator_budgets = budgets.clone();
    let mut seed = 0x1820_1820_1820_1820u64;
    for _ in 0..256 {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        allocator_budgets.push(usize::try_from(seed % (peak as u64 + 1)).unwrap());
    }
    budgets.sort_unstable();
    budgets.dedup();
    allocator_budgets.sort_unstable();
    allocator_budgets.dedup();
    let mut cases = 0;
    let mut memory = 0;
    for (policy, budgets) in [(true, budgets), (false, allocator_budgets)] {
        for budget in &budgets {
            let (accepted, measured) = run(policy, *budget);
            assert!(measured <= *budget);
            cases += 1;
            memory += usize::from(!accepted);
        }
    }
    assert!(run(true, 64 << 20).0, "the policy control must pass");
    assert!(run(false, 64 << 20).0, "the restored control must pass");
    println!(
        "{cases} budget cases: {memory} memory refusals, {} accepts; controls pass",
        cases - memory
    );
}
