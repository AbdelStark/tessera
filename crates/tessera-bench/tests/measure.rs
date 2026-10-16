// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The measurement harness of the microbenchmarks (RFC-0020 'Measurement protocol', 08 §6):
//! the statistics a regression gate compares, the wall time it measures, and the baseline it
//! writes.

use std::time::Duration;

use serde_json::Value;
use tessera_bench::{
    MICROBENCHMARKS_SCHEMA, Measurement, MicrobenchmarkReport, ModelError, Problem, Protocol,
    Statistics, measure,
};

/// The median of an even count is the mean of the two middle runs, the extremes are the
/// extremes, and the coefficient of variation is the sample standard deviation over the mean.
#[test]
fn statistics_are_those_the_gate_compares() {
    // Ten runs, unsorted: the middle two are 11 and 12.
    let runs = [14.0, 9.0, 10.0, 30.0, 12.0, 8.0, 13.0, 11.0, 12.5, 9.5];
    let s = Statistics::of(&runs).expect("statistics");
    assert_eq!((s.median, s.min, s.max), (11.5, 8.0, 30.0));
    let mean = runs.iter().sum::<f64>() / 10.0;
    let sd = (runs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / 9.0).sqrt();
    assert!((s.cv - sd / mean).abs() < 1e-15);
    assert!(s.noisy(), "one run at 30 is above the 3% flag");
    let steady = Statistics::of(&[100.0, 101.0, 99.0]).expect("statistics");
    assert_eq!(steady.median, 100.0);
    assert!(!steady.noisy());
    assert_eq!(Statistics::of(&[7.0]).expect("one run").cv, 0.0);
    assert_eq!(Statistics::of(&[0.0, 0.0]).expect("zero runs").cv, 0.0);
    let refused = |problem| {
        Err(ModelError::Input {
            name: "samples",
            problem,
        })
    };
    assert_eq!(Statistics::of(&[]), refused(Problem::Empty));
    assert_eq!(
        Statistics::of(&[1.0, f64::NAN]),
        refused(Problem::NotFinite)
    );
    assert_eq!(Statistics::of(&[1.0, -1.0]), refused(Problem::Negative));
}

/// `measure` times the run it is given, in wall time per unit of work: a run injected with
/// four times the sleep reports about four times the cost, and halving the units it is divided
/// by doubles it. Warm-up runs are not measured.
#[test]
fn measure_reports_wall_time_per_unit() {
    let protocol = Protocol {
        warmups: 1,
        iterations: 5,
    };
    let sleep = |ms| move || std::thread::sleep(Duration::from_millis(ms));
    let short = measure(protocol, 1.0, sleep(2)).expect("a measurement");
    let long = measure(protocol, 1.0, sleep(8)).expect("a measurement");
    // A sleep lasts at least its duration; schedulers add a little, never a factor of two.
    assert!(short.min >= 2e6 && long.min >= 8e6, "{short:?} {long:?}");
    let ratio = long.median / short.median;
    assert!(
        (2.5..=4.5).contains(&ratio),
        "4× the sleep reports {ratio:.2}× the cost"
    );
    let per_half = measure(protocol, 0.5, sleep(2)).expect("a measurement");
    assert!(
        per_half.median >= 4e6,
        "twice the cost per unit: {per_half:?}"
    );
    // Warm-up runs are the slow first ones; none of them reaches the statistics.
    let mut calls = 0;
    let warm = measure(
        Protocol {
            warmups: 3,
            iterations: 5,
        },
        1.0,
        || {
            calls += 1;
            std::thread::sleep(Duration::from_millis(if calls <= 3 { 20 } else { 1 }));
        },
    )
    .expect("a measurement");
    assert!(warm.max < 10e6, "a warm-up run was measured: {warm:?}");
    let mut runs = 0;
    measure(
        Protocol {
            warmups: 3,
            iterations: 10,
        },
        1.0,
        || runs += 1,
    )
    .expect("a measurement");
    assert_eq!(runs, 13, "3 warm-up runs and 10 measured ones");
    assert_eq!(
        measure(
            Protocol {
                warmups: 1,
                iterations: 0
            },
            1.0,
            || ()
        ),
        Err(ModelError::Input {
            name: "iterations",
            problem: Problem::NotPositive
        })
    );
    assert_eq!(
        measure(protocol, 0.0, || ()),
        Err(ModelError::Input {
            name: "units",
            problem: Problem::NotPositive
        })
    );
}

fn measurement(name: &str) -> Measurement {
    Measurement {
        name: name.to_owned(),
        unit: "product".to_owned(),
        units: 1 << 20,
        statistics: Statistics::of(&[3.0, 3.1, 2.9]).expect("statistics"),
    }
}

/// A baseline is a `tessera.microbenchmarks/1` object of finite numbers, whose runner class and
/// benchmark names are sanitized tokens, one measurement per name.
#[test]
fn baselines_are_schema_versioned_and_sanitized() {
    let report = MicrobenchmarkReport::new(
        "github-ubuntu-24.04",
        Protocol::RFC_0020,
        vec![measurement("field/fp2-mul"), measurement("ntt/fp")],
    )
    .expect("a report");
    let json: Value = serde_json::from_str(&report.to_json().expect("JSON")).expect("valid JSON");
    assert_eq!(json["schema"], MICROBENCHMARKS_SCHEMA);
    assert_eq!(json["runner_class"], "github-ubuntu-24.04");
    assert_eq!(json["protocol"]["iterations"], 10);
    assert_eq!(json["measurements"][1]["name"], "ntt/fp");
    assert_eq!(json["measurements"][0]["ns_per_unit"]["median"], 3.0);
    assert_eq!(json["measurements"][0]["noisy"], true);

    let malformed = |name| {
        Err(ModelError::Input {
            name,
            problem: Problem::Malformed,
        })
    };
    let new = |class: &str, names: &[&str]| {
        MicrobenchmarkReport::new(
            class,
            Protocol::RFC_0020,
            names.iter().map(|n| measurement(n)).collect(),
        )
        .map(|r| r.measurements.len())
    };
    assert_eq!(new("Ubuntu", &["ntt/fp"]), malformed("runner_class"));
    assert_eq!(new("a/b", &["ntt/fp"]), malformed("runner_class"));
    assert_eq!(new("", &["ntt/fp"]), malformed("runner_class"));
    assert_eq!(new(&"x".repeat(65), &["ntt/fp"]), malformed("runner_class"));
    assert_eq!(new("ci", &["ntt/fp", "ntt/fp"]), malformed("name"));
    assert_eq!(new("ci", &["ntt fp"]), malformed("name"));
    assert_eq!(
        new("ci", &[]),
        Err(ModelError::Input {
            name: "measurements",
            problem: Problem::Empty
        })
    );
    let mut infinite = report;
    infinite.measurements[0].statistics.max = f64::INFINITY;
    assert_eq!(infinite.to_json(), Err(ModelError::Output { name: "max" }));
}
