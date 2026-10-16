// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Measurements: the statistics of RFC-0020 'Measurement protocol' (warm-up runs, measured runs,
//! their median, extremes and coefficient of variation), and the `tessera.microbenchmarks/1`
//! report that stores a run of microbenchmarks as the baseline of 08 §6's regression gate.

use std::hint::black_box;
use std::time::Instant;

use serde_json::{Map, Number, Value};

use crate::error::{ModelError, Problem};

/// How many times a benchmark runs: unmeasured warm-up runs, then measured runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Protocol {
    /// Runs before the measured ones, which warm caches, allocators and clocks.
    pub warmups: u32,
    /// Measured runs.
    pub iterations: u32,
}

impl Protocol {
    /// RFC-0020's: 3 warm-up runs and 10 measured runs.
    pub const RFC_0020: Self = Self {
        warmups: 3,
        iterations: 10,
    };
}

/// The coefficient of variation above which RFC-0020 flags a measurement and repeats the run.
pub const NOISY_CV: f64 = 0.03;

/// The statistics of measured runs, each in nanoseconds per unit of work.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Statistics {
    /// The median run: the mean of the two middle runs for an even count.
    pub median: f64,
    /// The fastest run.
    pub min: f64,
    /// The slowest run.
    pub max: f64,
    /// The sample standard deviation over the mean; 0 for one run or a zero mean.
    pub cv: f64,
}

impl Statistics {
    /// The statistics of `samples`: at least one, each finite and non-negative.
    ///
    /// # Errors
    ///
    /// [`ModelError::Input`] for no sample, or a sample that is not finite or is negative.
    pub fn of(samples: &[f64]) -> Result<Self, ModelError> {
        if samples.is_empty() {
            return Err(ModelError::Input {
                name: "samples",
                problem: Problem::Empty,
            });
        }
        for &sample in samples {
            let problem = match (sample.is_finite(), sample < 0.0) {
                (false, _) => Problem::NotFinite,
                (true, true) => Problem::Negative,
                (true, false) => continue,
            };
            return Err(ModelError::Input {
                name: "samples",
                problem,
            });
        }
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len();
        let median = match n % 2 {
            1 => sorted[n / 2],
            _ => (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0,
        };
        let mean = sorted.iter().sum::<f64>() / n as f64;
        let cv = match (n, mean > 0.0) {
            (1, _) | (_, false) => 0.0,
            _ => {
                let variance =
                    sorted.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1) as f64;
                variance.sqrt() / mean
            }
        };
        Ok(Self {
            median,
            min: sorted[0],
            max: sorted[n - 1],
            cv,
        })
    }

    /// Whether the coefficient of variation exceeds [`NOISY_CV`], so that RFC-0020 repeats the
    /// run.
    pub fn noisy(&self) -> bool {
        self.cv > NOISY_CV
    }
}

/// Runs `run` `protocol.warmups` times, then `protocol.iterations` times measured: the
/// statistics of the measured runs' wall times, each divided by the `units` of work a run does,
/// in nanoseconds per unit. Each run's result goes through [`black_box`], so the work cannot be
/// optimized away.
///
/// # Errors
///
/// [`ModelError::Input`] for no measured run, or `units` not finite or not positive.
pub fn measure<T>(
    protocol: Protocol,
    units: f64,
    mut run: impl FnMut() -> T,
) -> Result<Statistics, ModelError> {
    if protocol.iterations == 0 {
        return Err(ModelError::Input {
            name: "iterations",
            problem: Problem::NotPositive,
        });
    }
    if !units.is_finite() || units <= 0.0 {
        return Err(ModelError::Input {
            name: "units",
            problem: match units.is_finite() {
                true => Problem::NotPositive,
                false => Problem::NotFinite,
            },
        });
    }
    for _ in 0..protocol.warmups {
        black_box(run());
    }
    let samples: Vec<f64> = (0..protocol.iterations)
        .map(|_| {
            let start = Instant::now();
            black_box(run());
            start.elapsed().as_secs_f64() * 1e9 / units
        })
        .collect();
    Statistics::of(&samples)
}

/// The report's schema identifier (03 §5).
pub const MICROBENCHMARKS_SCHEMA: &str = "tessera.microbenchmarks/1";

/// One benchmark's measurement.
#[derive(Clone, Debug, PartialEq)]
pub struct Measurement {
    /// The benchmark's stable name, such as `field/fp2-mul`, which a baseline is compared by:
    /// lowercase ASCII letters, digits, `-`, `_`, `.` and `/`.
    pub name: String,
    /// The unit of work the statistics are per, such as `product`: lowercase ASCII letters and
    /// spaces.
    pub unit: String,
    /// The units of work each run does.
    pub units: u64,
    /// The statistics, in nanoseconds per unit.
    pub statistics: Statistics,
}

/// One run of microbenchmarks on one runner class: the baseline that 08 §6's gate compares a
/// candidate with, benchmark by benchmark and median by median, on the same runner class.
#[derive(Clone, Debug, PartialEq)]
pub struct MicrobenchmarkReport {
    /// The runner class, such as `github-ubuntu-24.04`: lowercase ASCII letters, digits, `-`,
    /// `_` and `.`.
    pub runner_class: String,
    /// The operating system, `std::env::consts::OS`.
    pub os: String,
    /// The architecture, `std::env::consts::ARCH`.
    pub arch: String,
    /// How many times each benchmark ran.
    pub protocol: Protocol,
    /// The measurements, in the order they ran.
    pub measurements: Vec<Measurement>,
}

/// Whether `text` is 1 to 64 bytes of lowercase ASCII letters, digits and `extra`.
fn token(text: &str, extra: &[u8]) -> bool {
    (1..=64).contains(&text.len())
        && text
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || extra.contains(&b))
}

impl MicrobenchmarkReport {
    /// A report of `measurements` on `runner_class`, for this process's operating system and
    /// architecture.
    ///
    /// # Errors
    ///
    /// [`ModelError::Input`] for a runner class, a benchmark name or a unit outside its
    /// alphabet or longer than 64 bytes, two benchmarks of one name, or no measurement.
    pub fn new(
        runner_class: &str,
        protocol: Protocol,
        measurements: Vec<Measurement>,
    ) -> Result<Self, ModelError> {
        let malformed = |name| {
            Err(ModelError::Input {
                name,
                problem: Problem::Malformed,
            })
        };
        if !token(runner_class, b"-_.") {
            return malformed("runner_class");
        }
        if measurements.is_empty() {
            return Err(ModelError::Input {
                name: "measurements",
                problem: Problem::Empty,
            });
        }
        for (k, m) in measurements.iter().enumerate() {
            if !token(&m.name, b"-_./") || measurements[..k].iter().any(|e| e.name == m.name) {
                return malformed("name");
            }
            if !token(&m.unit, b" ") {
                return malformed("unit");
            }
        }
        Ok(Self {
            runner_class: runner_class.to_owned(),
            os: std::env::consts::OS.to_owned(),
            arch: std::env::consts::ARCH.to_owned(),
            protocol,
            measurements,
        })
    }

    /// The report as a `tessera.microbenchmarks/1` JSON object, two-space indented, its keys
    /// sorted. Every number is finite.
    ///
    /// # Errors
    ///
    /// [`ModelError::Output`] for a statistic that is not finite, which [`measure`] never
    /// produces but a report built by hand can hold.
    pub fn to_json(&self) -> Result<String, ModelError> {
        let number = |name: &'static str, value: f64| {
            Number::from_f64(value)
                .map(Value::Number)
                .ok_or(ModelError::Output { name })
        };
        let measurements = self
            .measurements
            .iter()
            .map(|m| {
                let s = &m.statistics;
                let mut statistics = Map::new();
                statistics.insert("median".to_owned(), number("median", s.median)?);
                statistics.insert("min".to_owned(), number("min", s.min)?);
                statistics.insert("max".to_owned(), number("max", s.max)?);
                statistics.insert("cv".to_owned(), number("cv", s.cv)?);
                let mut entry = Map::new();
                entry.insert("name".to_owned(), Value::from(m.name.as_str()));
                entry.insert("unit".to_owned(), Value::from(m.unit.as_str()));
                entry.insert("units".to_owned(), Value::from(m.units));
                entry.insert("ns_per_unit".to_owned(), Value::Object(statistics));
                entry.insert("noisy".to_owned(), Value::from(s.noisy()));
                Ok(Value::Object(entry))
            })
            .collect::<Result<Vec<_>, ModelError>>()?;
        let mut protocol = Map::new();
        protocol.insert("warmups".to_owned(), Value::from(self.protocol.warmups));
        protocol.insert(
            "iterations".to_owned(),
            Value::from(self.protocol.iterations),
        );
        let mut report = Map::new();
        report.insert("schema".to_owned(), Value::from(MICROBENCHMARKS_SCHEMA));
        report.insert(
            "runner_class".to_owned(),
            Value::from(self.runner_class.as_str()),
        );
        report.insert("os".to_owned(), Value::from(self.os.as_str()));
        report.insert("arch".to_owned(), Value::from(self.arch.as_str()));
        report.insert("protocol".to_owned(), Value::Object(protocol));
        report.insert("measurements".to_owned(), Value::Array(measurements));
        serde_json::to_string_pretty(&Value::Object(report))
            .map_err(|_| ModelError::Output { name: "report" })
    }
}
