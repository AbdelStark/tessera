// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The driver's deadline must interrupt opening work before its scratch can fail (SP-8).
use cap::Cap;
use std::alloc::System;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use tv_codec::{DecodeLimits, WeightCommitmentPublic};
use tv_core::claims::ClaimList;
use tv_core::observe::{CheckEvent, CheckKind, Observer};
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

struct OpeningBudget {
    expire: bool,
    budget: Option<usize>,
    baseline: Option<usize>,
    padding: Vec<u8>,
}
impl Observer for OpeningBudget {
    fn check_completed(&mut self, event: CheckEvent) {
        if event.kind != CheckKind::Queries {
            return;
        }
        assert!(self.baseline.is_none());
        self.padding = vec![0; ALLOCATOR.max_allocated() - ALLOCATOR.allocated()];
        let baseline = ALLOCATOR.allocated();
        assert_eq!(baseline, ALLOCATOR.max_allocated());
        self.baseline = Some(baseline);
        if let Some(budget) = self.budget {
            ALLOCATOR
                .set_limit(baseline.checked_add(budget).unwrap())
                .unwrap();
        }
        if self.expire {
            std::thread::sleep(Duration::from_millis(2100));
        }
    }
}

fn worker(expire: bool, budget: Option<usize>) {
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
    if expire {
        cfg.limits.max_wall_time = Duration::from_secs(2);
    }
    let mut observer = OpeningBudget {
        expire,
        budget,
        baseline: None,
        padding: Vec::new(),
    };
    let verdict = tv_core::verify_non_interactive_bytes_observed(
        &claims,
        &statement,
        &bytes,
        &cfg,
        &mut observer,
    );
    ALLOCATOR.set_limit(usize::MAX).unwrap();
    let peak = ALLOCATOR.max_allocated() - observer.baseline.expect("must reach opening work");
    std::hint::black_box(&observer.padding);
    if let Some(budget) = budget {
        assert!(peak <= budget);
    }
    let outcome = match verdict {
        Verdict::Accept(_) => "accept",
        Verdict::Inconclusive(e) if e.code == ErrorCode::Time => "time",
        Verdict::Inconclusive(e) if e.code == ErrorCode::Memory => "memory",
        other => panic!("unexpected verdict: {other:?}"),
    };
    println!("{outcome} {peak}");
}

fn run(expire: bool, budget: Option<usize>) -> (String, usize) {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--worker", if expire { "expired" } else { "active" }])
        .arg(budget.map_or_else(|| "unlimited".into(), |n| n.to_string()))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let started = Instant::now();
    while child.try_wait().unwrap().is_none() {
        if started.elapsed() > Duration::from_secs(30) {
            let _ = child.kill();
            child.wait().unwrap();
            panic!("opening deadline worker exceeded 30 seconds");
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = std::str::from_utf8(&output.stdout).unwrap();
    let mut fields = text.split_whitespace();
    let outcome = fields.next().unwrap().to_owned();
    let peak = fields.next().unwrap().parse().unwrap();
    assert!(fields.next().is_none());
    (outcome, peak)
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    if let [_, flag, mode, budget] = args.as_slice() {
        assert_eq!(flag, "--worker");
        assert!(matches!(mode.as_str(), "expired" | "active"));
        worker(
            mode == "expired",
            if budget == "unlimited" {
                None
            } else {
                Some(budget.parse().unwrap())
            },
        );
        return;
    }
    assert_eq!(args.len(), 1);
    assert_eq!(run(false, None).0, "accept");
    let (outcome, budget) = run(true, None);
    assert_eq!(outcome, "time");
    assert!(budget > 0);
    assert_eq!(run(true, Some(budget)).0, "time");
    // Independent control: this exact cap cannot complete ordinary opening work. Thus
    // calibrating the cap from a run that ignores cancellation cannot make a false pass.
    assert_eq!(run(false, Some(budget)).0, "memory");
    assert_eq!(run(false, None).0, "accept");
    println!(
        "opening work: deadline beats scratch shortage at {budget} bytes; active cap refuses memory; restored control accepts"
    );
}
