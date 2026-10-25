// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The verifier's audit plan (RFC-0016 'Calculators') at its edges, against figures computed
//! here from the formulas: the lowest later rate, a rate of 1 and a bucket without violations,
//! the overhead and the capped target rates, and each refusal in its order. The example plan is
//! `tv-cli`'s test.

use tv_core::audit::{AuditPolicy, RateBucket};
use tv_core::audit_plan::{AuditPlan, CostModel, PlanError, TrafficBucket};
use tv_core::claims::ParamSetId;

fn policy(buckets: &[(u32, u64)]) -> AuditPolicy {
    AuditPolicy::new(
        1,
        buckets
            .iter()
            .map(|&(max_tokens, threshold)| RateBucket {
                max_tokens,
                threshold,
            })
            .collect(),
        1,
        2,
        1,
        ParamSetId::new("live-64/1").unwrap(),
    )
    .unwrap()
}

fn traffic(rows: &[(u64, u32, u64)]) -> Vec<TrafficBucket> {
    rows.iter()
        .map(|&(requests, tokens, violations)| TrafficBucket {
            requests,
            tokens,
            violations,
        })
        .collect()
}

fn close(a: f64, b: f64) {
    assert!(
        (a - b).abs() <= 1e-12 * a.abs().max(b.abs()),
        "{a} against {b}"
    );
}

const HALF: u64 = 1 << 63;
const QUARTER: u64 = 1 << 62;

#[test]
fn detection_takes_the_lowest_later_rate_and_in_place_its_own() {
    // Rates 1/4, 1/2, 1/4: the first two buckets are detected at 1/4 at best.
    let p = policy(&[(10, QUARTER), (20, HALF), (30, QUARTER)]);
    let t = traffic(&[(100, 5, 3), (100, 15, 2), (100, 25, 1)]);
    let plan = AuditPlan::new(&p, &t, None).unwrap();
    let rates: Vec<(f64, f64)> = plan
        .buckets
        .iter()
        .map(|b| (b.rate, b.lowest_later_rate))
        .collect();
    assert_eq!(rates, [(0.25, 0.25), (0.5, 0.25), (0.25, 0.25)]);
    let padded = 1.0 - 0.75f64.powi(6);
    let in_place = 1.0 - 0.75f64.powi(3) * 0.5f64.powi(2) * 0.75;
    close(plan.detection, padded);
    close(plan.detection_in_place, in_place);
    close(plan.buckets[1].detection, 1.0 - 0.75f64.powi(2));
    close(plan.buckets[1].detection_in_place, 1.0 - 0.5f64.powi(2));
    close(plan.expected_selected, 100.0 * (0.25 + 0.5 + 0.25));
    assert!(plan.cost.is_none());
    let json = plan.to_json().unwrap();
    assert!(json.contains("\"overhead\":null") && json.contains("\"mu\":null"));
}

#[test]
fn a_rate_of_one_detects_surely_and_no_violation_detects_nothing() {
    let p = policy(&[(10, 0), (20, u64::MAX)]);
    let t = traffic(&[(5, 10, 0), (5, 11, 4)]);
    let plan = AuditPlan::new(&p, &t, None).unwrap();
    // No violation in the first bucket: 0, not 0 · ∞; the rate 1 bucket: 1.
    assert_eq!(plan.buckets[0].detection, 0.0);
    assert!(plan.buckets[0].detection.is_sign_positive());
    assert_eq!(plan.buckets[1].rate, 1.0);
    assert_eq!(plan.buckets[1].detection, 1.0);
    assert_eq!(plan.detection, 1.0);
    // A rate of 1 with no violation: 0, not 0 · ln(0).
    let sure = AuditPlan::new(&policy(&[(10, u64::MAX)]), &traffic(&[(5, 10, 0)]), None).unwrap();
    assert_eq!(sure.buckets[0].detection, 0.0);
    assert_eq!(sure.detection, 0.0);
    assert!(sure.to_json().is_ok());
    // A rate of 0 everywhere detects nothing, and the zeros print without a sign.
    let none = AuditPlan::new(&policy(&[(10, 0)]), &traffic(&[(5, 10, 5)]), None).unwrap();
    assert_eq!(none.detection, 0.0);
    assert!(!none.to_json().unwrap().contains("-0.0"));
}

#[test]
fn the_overhead_and_the_target_rates_follow_mu() {
    let p = policy(&[(10, HALF), (100, QUARTER)]);
    let t = traffic(&[(4, 10, 0), (2, 50, 0)]);
    let cost = CostModel {
        source_defaults: false,
        eta: 0.5,
        target_overhead: 0.1,
        mu: vec![(10, 0.1), (50, 4.0)],
    };
    let plan = AuditPlan::new(&p, &t, Some(&cost)).unwrap();
    let c = plan.cost.unwrap();
    // Σ n·L·s·μ·η / Σ n·L = (40·0.5·0.1 + 100·0.25·4)·0.5 / 140.
    close(
        c.overhead,
        (40.0 * 0.5 * 0.1 + 100.0 * 0.25 * 4.0) * 0.5 / 140.0,
    );
    assert!(!c.source_defaults);
    // O/(μ·η): 0.1/0.05 = 2 is capped at 1; 0.1/2 = 0.05.
    assert_eq!(plan.buckets[0].target_rate, Some(1.0));
    close(plan.buckets[1].target_rate.unwrap(), 0.05);
    assert_eq!(plan.buckets[1].mu, Some(4.0));
}

#[test]
fn each_ill_formed_profile_or_cost_model_is_refused_in_order() {
    let p = policy(&[(10, HALF), (20, HALF)]);
    let good = [(1, 5, 0), (1, 15, 0)];
    let refuse = |rows: &[(u64, u32, u64)], cost: Option<&CostModel>| {
        AuditPlan::new(&p, &traffic(rows), cost).err()
    };
    assert_eq!(refuse(&good, None), None);
    // One entry per bucket, and a policy with buckets.
    assert_eq!(refuse(&good[..1], None), Some(PlanError::Buckets));
    assert_eq!(
        refuse(&[(1, 5, 0), (1, 15, 0), (1, 25, 0)], None),
        Some(PlanError::Buckets)
    );
    assert_eq!(
        AuditPlan::new(&policy(&[]), &[], None).err(),
        Some(PlanError::Buckets)
    );
    // Mean lengths: 0, past the bucket, at the previous bucket's bound.
    for (rows, bucket) in [
        ([(1, 0, 0), (1, 15, 0)], 0),
        ([(1, 11, 0), (1, 15, 0)], 0),
        ([(1, 5, 0), (1, 10, 0)], 1),
        ([(1, 5, 0), (1, 21, 0)], 1),
    ] {
        assert_eq!(refuse(&rows, None), Some(PlanError::Tokens { bucket }));
    }
    // The edges pass: the bucket's own bound, one above the previous.
    assert_eq!(refuse(&[(1, 10, 0), (1, 11, 0)], None), None);
    assert_eq!(refuse(&[(1, 1, 1), (0, 20, 0)], None), None);
    // More violations than requests; no request at all.
    assert_eq!(
        refuse(&[(1, 5, 2), (1, 15, 0)], None),
        Some(PlanError::Violations { bucket: 0 })
    );
    assert_eq!(
        refuse(&[(0, 5, 0), (0, 15, 0)], None),
        Some(PlanError::NoRequests)
    );
    // The order: the lengths before the violations.
    assert_eq!(
        refuse(&[(1, 5, 2), (1, 25, 0)], None),
        Some(PlanError::Tokens { bucket: 1 })
    );
    // The cost model.
    let cost = |mu: Vec<(u64, f64)>, eta: f64, target_overhead: f64| CostModel {
        source_defaults: true,
        eta,
        target_overhead,
        mu,
    };
    let ok = cost(vec![(5, 1.0), (15, 2.0)], 0.3, 0.1);
    assert_eq!(refuse(&good, Some(&ok)), None);
    for (model, error) in [
        (cost(vec![(5, 1.0)], 0.3, 0.1), PlanError::Contexts),
        (
            cost(vec![(15, 1.0), (5, 2.0)], 0.3, 0.1),
            PlanError::Contexts,
        ),
        (
            cost(vec![(5, 1.0), (16, 2.0)], 0.3, 0.1),
            PlanError::Contexts,
        ),
        (
            cost(vec![(5, 1.0), (15, 2.0), (25, 1.0)], 0.3, 0.1),
            PlanError::Contexts,
        ),
        (
            cost(vec![(5, 0.0), (15, 2.0)], 0.3, 0.1),
            PlanError::Mu { bucket: 0 },
        ),
        (
            cost(vec![(5, 1.0), (15, f64::NAN)], 0.3, 0.1),
            PlanError::Mu { bucket: 1 },
        ),
        (
            cost(vec![(5, 1.0), (15, f64::INFINITY)], 0.3, 0.1),
            PlanError::Mu { bucket: 1 },
        ),
        (cost(vec![(5, 1.0), (15, 2.0)], 0.0, 0.1), PlanError::Eta),
        (
            cost(vec![(5, 1.0), (15, 2.0)], 0.3, 0.0),
            PlanError::TargetOverhead,
        ),
        (
            cost(vec![(5, 1.0), (15, 2.0)], 0.3, f64::NAN),
            PlanError::TargetOverhead,
        ),
    ] {
        assert_eq!(refuse(&good, Some(&model)), Some(error), "{model:?}");
    }
    assert_eq!(PlanError::NoRequests.code().id(), "ERR-USAGE");
}
