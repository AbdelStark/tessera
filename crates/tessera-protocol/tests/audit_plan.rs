// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0016 'Calculators', through [`AuditPlan`]:
//! - the source's §8 table, reproduced from its `μ` values and `η = 0.3`: each context's rate for
//!   10% amortized overhead, and the detection of 100 violating requests at it;
//! - the closed-form detection against a Monte Carlo run of the selection rule itself
//!   ([`AuditPolicy::selects`]), within 0.5 percentage points, with every violation left in its
//!   bucket and with every violation padded into the later bucket of lowest rate;
//! - the lowest later rate over a policy whose rates do not decrease;
//! - the amortized overhead, and every input the calculators refuse.

use tessera_codec::Ascii;
use tessera_hash::ParamSetId;
use tessera_protocol::{AuditPlan, AuditPolicy, BucketTraffic, PlanError, RateBucket, RequestKey};

/// A policy of `(max_tokens, rate)` buckets, the rates decimal strings.
fn policy(buckets: &[(u32, &str)]) -> AuditPolicy {
    let buckets = buckets
        .iter()
        .map(|&(max_tokens, rate)| RateBucket {
            max_tokens,
            threshold: RateBucket::threshold_of(rate).expect("a rate"),
        })
        .collect();
    AuditPolicy::new(
        86_400_000_000_000,
        buckets,
        16_384,
        262_144,
        3_600_000_000_000,
        ParamSetId(Ascii::new("live-64/1").expect("ASCII")),
    )
    .expect("a policy")
}

/// `n` requests of `tokens` tokens, `violations` of them violating.
fn traffic(requests: u64, tokens: u64, violations: u64) -> BucketTraffic {
    BucketTraffic {
        requests,
        tokens,
        violations,
    }
}

/// SplitMix64, the Monte Carlo run's generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
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

/// The fraction of `trials` samples that select at least one violating request: in each, a fresh
/// seed, and fresh keys for `violations[b]` requests of `lengths[b]` tokens.
fn monte_carlo(
    policy: &AuditPolicy,
    violations: &[u64],
    lengths: &[u64],
    trials: u32,
    rng: &mut Rng,
) -> f64 {
    let mut detected = 0u32;
    for _ in 0..trials {
        let seed = rng.bytes::<32>();
        let mut hit = false;
        for (&v, &length) in violations.iter().zip(lengths) {
            for _ in 0..v {
                let key = RequestKey {
                    session_id: rng.bytes(),
                    request_id: rng.next(),
                };
                hit |= policy.selects(&seed, &key, length);
            }
        }
        detected += u32::from(hit);
    }
    f64::from(detected) / f64::from(trials)
}

/// The source's §8 table: `μ` 3.2, 6.5, 18.8 and 88.7 at 1K, 4K, 16K and 128K tokens, `η = 0.3`,
/// 10% amortized overhead. Its rates are 10.4%, 5.1%, 1.8% and 0.38%, and 100 violating
/// requests are detected with probability ≈100%, 99.5%, 83.3% and 31.4%.
#[test]
fn the_sources_sampling_table_is_reproduced_from_its_inputs() {
    let contexts = [1_024u32, 4_096, 16_384, 131_072];
    let mu = [3.2, 6.5, 18.8, 88.7];
    let (eta, target) = (0.3, 0.1);
    // The target rates, from any policy: they depend on μ, η and the target only.
    let probe = policy(&[
        (1_024, "0.1"),
        (4_096, "0.1"),
        (16_384, "0.1"),
        (131_072, "0.1"),
    ]);
    let flat = [
        traffic(1, 1_024, 0),
        traffic(1, 4_096, 0),
        traffic(1, 16_384, 0),
        traffic(1, 131_072, 0),
    ];
    let plan = AuditPlan::new(&probe, &flat, &mu, eta, target).expect("a plan");
    let rates: Vec<f64> = plan.buckets.iter().map(|b| b.target_rate).collect();
    // As the table prints them: one decimal, and two for the 128K row.
    let shown: Vec<String> = rates
        .iter()
        .enumerate()
        .map(|(b, r)| match b {
            3 => format!("{:.2}", 100.0 * r),
            _ => format!("{:.1}", 100.0 * r),
        })
        .collect();
    assert_eq!(shown, ["10.4", "5.1", "1.8", "0.38"]);

    // The policy that samples each context at its target rate, and 100 violating requests in
    // each, left in their bucket.
    let decimals: Vec<String> = rates.iter().map(|r| format!("{r:.19}")).collect();
    let buckets: Vec<(u32, &str)> = contexts
        .iter()
        .zip(&decimals)
        .map(|(&c, d)| (c, d.as_str()))
        .collect();
    let at_target = policy(&buckets);
    for (b, &context) in contexts.iter().enumerate() {
        let mut one = flat;
        one[b] = traffic(1_000_000, u64::from(context), 100);
        let plan = AuditPlan::new(&at_target, &one, &mu, eta, target).expect("a plan");
        let detection = 100.0 * plan.buckets[b].detection_in_place;
        let expected = ["100.0", "99.5", "83.3", "31.4"][b];
        assert_eq!(format!("{detection:.1}"), expected, "{context} tokens");
        // At its target rate, a context's traffic alone costs the target overhead.
        assert!(
            (plan.overhead - target).abs() < 1e-12,
            "{context}: {}",
            plan.overhead
        );
    }
}

/// The closed form against the selection rule, in place and padded, within 0.5 points.
#[test]
fn closed_form_detection_matches_a_monte_carlo_run_of_the_selection_rule() {
    // RFC-0016's example rates, and violations that leave each closed form well inside (0, 1).
    let p = policy(&[(1_024, "0.1"), (16_384, "0.02"), (131_072, "0.004")]);
    let violations = [6u64, 20, 60];
    let profile = [
        traffic(800_000, 700, violations[0]),
        traffic(150_000, 9_000, violations[1]),
        traffic(50_000, 60_000, violations[2]),
    ];
    let plan = AuditPlan::new(&p, &profile, &[3.2, 18.8, 88.7], 0.3, 0.1).expect("a plan");
    let mut rng = Rng(0x0141_0001);
    let trials = 100_000;

    // Every violation in its own bucket: 1 − 0.9^6·0.98^20·0.996^60.
    let in_place = monte_carlo(&p, &violations, &[700, 9_000, 60_000], trials, &mut rng);
    let closed = 1.0 - 0.9f64.powi(6) * 0.98f64.powi(20) * 0.996f64.powi(60);
    assert!((plan.detection_in_place - closed).abs() < 1e-12);
    assert!(
        (in_place - plan.detection_in_place).abs() < 0.005,
        "in place: Monte Carlo {in_place}, closed form {}",
        plan.detection_in_place
    );

    // Every violation padded into the 128K bucket, the lowest rate after each: 1 − 0.996^86.
    let padded = monte_carlo(&p, &violations, &[100_000; 3], trials, &mut rng);
    let closed = 1.0 - 0.996f64.powi(86);
    assert!((plan.detection - closed).abs() < 1e-12);
    assert!(
        (padded - plan.detection).abs() < 0.005,
        "padded: Monte Carlo {padded}, closed form {}",
        plan.detection
    );
    // Padding costs the policy most of its detection.
    assert!(plan.detection_in_place > 0.7 && plan.detection < 0.3);
}

/// `s̲_b = min_{c ≥ b} s_c` when the rates do not decrease: a violation of the first bucket pads
/// into the second, not the third, and the last bucket keeps its own rate.
#[test]
fn the_lowest_later_rate_looks_past_a_higher_rate() {
    let p = policy(&[(1_024, "0.1"), (16_384, "0.02"), (131_072, "0.05")]);
    let profile = [
        traffic(1_000, 1_000, 10),
        traffic(1_000, 10_000, 10),
        traffic(1_000, 100_000, 10),
    ];
    let plan = AuditPlan::new(&p, &profile, &[3.2, 18.8, 88.7], 0.3, 0.1).expect("a plan");
    let lowest: Vec<f64> = plan.buckets.iter().map(|b| b.lowest_later_rate).collect();
    let rates: Vec<f64> = plan.buckets.iter().map(|b| b.rate).collect();
    for (got, want) in lowest.iter().zip([0.02, 0.02, 0.05]) {
        assert!((got - want).abs() < 1e-15, "{lowest:?}");
    }
    for (got, want) in rates.iter().zip([0.1, 0.02, 0.05]) {
        assert!((got - want).abs() < 1e-15, "{rates:?}");
    }
    let closed = 1.0 - 0.98f64.powi(20) * 0.95f64.powi(10);
    assert!((plan.detection - closed).abs() < 1e-12);
    let mut rng = Rng(0x0141_0002);
    let padded = monte_carlo(
        &p,
        &[10, 10, 10],
        &[10_000, 10_000, 100_000],
        100_000,
        &mut rng,
    );
    assert!(
        (padded - plan.detection).abs() < 0.005,
        "{padded} {}",
        plan.detection
    );
}

/// A rate of 1 detects every violation, a rate of 0 none, and no violation is never detected.
#[test]
fn certain_and_impossible_detection() {
    let p = policy(&[(1_024, "1"), (16_384, "0")]);
    let mu = [3.2, 18.8];
    let only_first = [traffic(10, 1_000, 3), traffic(10, 10_000, 0)];
    let plan = AuditPlan::new(&p, &only_first, &mu, 0.3, 0.1).expect("a plan");
    assert_eq!(plan.buckets[0].detection_in_place, 1.0);
    // Padded into the second bucket, at rate 0, the same violations escape.
    assert_eq!(plan.buckets[0].detection, 0.0);
    assert_eq!(plan.detection, 0.0);
    assert_eq!(plan.detection_in_place, 1.0);
    let none = [traffic(10, 1_000, 0), traffic(10, 10_000, 0)];
    let plan = AuditPlan::new(&p, &none, &mu, 0.3, 0.1).expect("a plan");
    assert_eq!((plan.detection, plan.detection_in_place), (0.0, 0.0));
}

/// `Σ n_b·L_b·s_b·μ_b·η / Σ n_b·L_b`, the expected selections `Σ n_b·s_b`, and target rates
/// capped at 1.
#[test]
fn amortized_overhead_expected_selections_and_target_rates() {
    let p = policy(&[(1_024, "0.1"), (16_384, "0.02")]);
    let profile = [traffic(900, 1_000, 0), traffic(100, 10_000, 0)];
    let plan = AuditPlan::new(&p, &profile, &[3.2, 18.8], 0.3, 0.1).expect("a plan");
    let cost = 900.0 * 1_000.0 * 0.1 * 3.2 * 0.3 + 100.0 * 10_000.0 * 0.02 * 18.8 * 0.3;
    let overhead = cost / (900.0 * 1_000.0 + 100.0 * 10_000.0);
    assert!(
        (plan.overhead - overhead).abs() < 1e-12,
        "{}",
        plan.overhead
    );
    assert!((plan.expected_selected - (90.0 + 2.0)).abs() < 1e-9);
    // A cheap context's target rate is capped at 1.
    let plan = AuditPlan::new(&p, &profile, &[0.1, 18.8], 0.3, 0.1).expect("a plan");
    assert_eq!(plan.buckets[0].target_rate, 1.0);
}

#[test]
fn the_calculators_refuse_ill_formed_inputs() {
    let p = policy(&[(1_024, "0.1"), (16_384, "0.02")]);
    let good = [traffic(10, 1_000, 1), traffic(10, 10_000, 1)];
    let mu = [3.2, 18.8];
    assert!(AuditPlan::new(&p, &good, &mu, 0.3, 0.1).is_ok());
    assert_eq!(
        AuditPlan::new(&p, &good[..1], &mu, 0.3, 0.1),
        Err(PlanError::Buckets {
            policy: 2,
            given: 1
        })
    );
    assert_eq!(
        AuditPlan::new(&p, &good, &mu[..1], 0.3, 0.1),
        Err(PlanError::Buckets {
            policy: 2,
            given: 1
        })
    );
    // A mean length outside its bucket: 0, past the bucket, or not above the bucket before.
    for (bucket, tokens) in [(0, 0), (0, 1_025), (1, 1_024), (1, 16_385)] {
        let mut t = good;
        t[bucket].tokens = tokens;
        assert_eq!(
            AuditPlan::new(&p, &t, &mu, 0.3, 0.1),
            Err(PlanError::Tokens { bucket }),
            "{tokens}"
        );
    }
    // The bounds themselves are in the bucket.
    let edges = [traffic(10, 1, 0), traffic(10, 16_384, 0)];
    assert!(AuditPlan::new(&p, &edges, &mu, 0.3, 0.1).is_ok());
    let mut t = good;
    t[1].violations = 11;
    assert_eq!(
        AuditPlan::new(&p, &t, &mu, 0.3, 0.1),
        Err(PlanError::Violations { bucket: 1 })
    );
    let idle = [traffic(0, 1_000, 0), traffic(0, 10_000, 0)];
    assert_eq!(
        AuditPlan::new(&p, &idle, &mu, 0.3, 0.1),
        Err(PlanError::NoTraffic)
    );
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            AuditPlan::new(&p, &good, &mu, bad, 0.1),
            Err(PlanError::Input { name: "η" })
        );
        assert_eq!(
            AuditPlan::new(&p, &good, &mu, 0.3, bad),
            Err(PlanError::Input {
                name: "the target overhead"
            })
        );
        assert_eq!(
            AuditPlan::new(&p, &good, &[3.2, bad], 0.3, 0.1),
            Err(PlanError::Input { name: "μ" })
        );
    }
}
