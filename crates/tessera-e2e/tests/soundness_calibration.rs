// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Soundness calibration over the test field `F65537` (07 §4; RFC-0010, RFC-0008 and RFC-0072
//! 'Testing Strategy'): cheating provers that guess one challenge succeed at the analytic rate,
//! `d/|F|` for a lie in one sumcheck round, up to `(N + T)/|F|` for lookups outside their table,
//! and `1 − (1 − 1/|F|)^n ≤ n/|F|` for a non-Boolean bit in stage 1's booleanity zero-check of
//! `n` variables, which passes exactly when the zero-check point zeroes `eq(z, x_0)`.
//!
//! Each setting runs a cheating prover on the real engine (`tessera-sumcheck`, `tessera-logup`)
//! and RFC-0004's transcript, and checks every proof with a verifier written here from RFC-0010
//! 'Sumcheck messages' and 'Tree levels' and RFC-0008 'Fraction tree': `tv-core` checks
//! Goldilocks proofs only. The cheating provers' acceptance depends on one challenge alone, and
//! is exact: the lie is accepted exactly when `r_0` is one of the `d` roots of the round's
//! difference polynomial, the bad lookups exactly when `α` is one of the `N + T − 1` roots of
//! their identity, the non-Boolean bit exactly when a coordinate of the zero-check point is its
//! root. Every trial checks that the verdict is this event. Over many trials, the rate
//! must contain the roots' share of the field in its 99% Clopper–Pearson interval, and the lower
//! end must not exceed the analytic bound.
//!
//! The quick tests below run with `cargo test`; the measurement is ignored there and runs in the
//! nightly workflow, in the dev profile, whose overflow checks stay on:
//!
//! ```text
//! cargo test --locked -p tessera-e2e --test soundness_calibration -- --ignored --nocapture
//! ```
//!
//! `docs/testing/soundness-calibration.md` describes the design and the latest results.

#![allow(clippy::print_stderr)]

mod calibration;

use std::time::Instant;

use calibration::booleanity::NonBoolean;
use calibration::interval::{binomial_cdf, clopper_pearson};
use calibration::logup::Lookups;
use calibration::sumcheck::RoundLie;
use calibration::{F, FIELD, Scripted, SmallTranscript, SplitMix64, count_accepted};
use tessera_field::{Field, PrimeField};

/// The confidence of every interval (07 §4).
const CONFIDENCE: f64 = 0.99;

/// The accepts each setting expects: `⌈200·|F|/K⌉` trials for `K` accepting challenges. The
/// 99% interval's half-width is then about 18% of the rate, so that an analytic rate off by a
/// factor of 2 falls outside it at every setting.
const EXPECTED_ACCEPTS: u64 = 200;

/// The trials that expect [`EXPECTED_ACCEPTS`] accepts from `roots` accepting challenges.
fn trials(roots: usize) -> u64 {
    (EXPECTED_ACCEPTS * FIELD).div_ceil(roots as u64)
}

/// A lie in round 0 of a sumcheck of degree `degree` on `vars` variables.
struct LieSetting {
    degree: usize,
    vars: usize,
}

/// Lookups outside their table: `lookups` keys none of the `rows` rows hold.
struct LookupSetting {
    lookups: usize,
    rows: usize,
}

/// The sumcheck settings: degrees 1, 3 and 7, the smallest, a tree level's and the largest that
/// RFC-0014 allows, each on 2 variables.
const LIES: [LieSetting; 3] = [
    LieSetting { degree: 1, vars: 2 },
    LieSetting { degree: 3, vars: 2 },
    LieSetting { degree: 7, vars: 2 },
];

/// The LogUp settings: tables of `2^4`, `2^8 − 1` and `2^10 − 1` rows, with 1, 2 and 3 lookups
/// outside them; the lookup trees have heights 0, 1 and 2, the last padded with row 0.
const LOOKUPS: [LookupSetting; 3] = [
    LookupSetting {
        lookups: 1,
        rows: 16,
    },
    LookupSetting {
        lookups: 2,
        rows: 255,
    },
    LookupSetting {
        lookups: 3,
        rows: 1023,
    },
];

/// The booleanity settings: one non-Boolean bit among the bits of a zero-check on 2 and on 4
/// variables, whose rates `n/|F|` the interval tells apart.
const ZERO_CHECKS: [usize; 2] = [2, 4];

/// A setting's seed: the statement digest of its sessions, and its instances' randomness.
fn seed(kind: u64, index: usize) -> u64 {
    0x0104_0000 + (kind << 8) + index as u64
}

/// A lie of degree `d` differs from the honest round polynomial by `δ = Π_k (X − (P − k))`,
/// `k = 1, …, d`: `d` roots at the top of the field, next to where the challenge sampler starts
/// rejecting, and `δ(0) + δ(1) ≠ 0`, so that the claim it supports is false.
#[test]
fn a_lying_round_has_its_degree_in_roots_at_the_top_of_the_field() {
    for degree in 1..=7 {
        let lie = RoundLie::new(degree, 2);
        let roots: Vec<u64> = lie.roots().iter().map(F::to_canonical_u64).collect();
        let expected: Vec<u64> = (1..=degree as u64).map(|k| FIELD - k).collect();
        assert_eq!(roots, expected, "degree {degree}");
        // δ vanishes there and nowhere else in the field.
        let zeros = (0..FIELD)
            .filter(|&x| lie.delta(F::from_u64_reduced(x)) == F::ZERO)
            .count();
        assert_eq!(zeros, degree, "degree {degree}: δ has d roots in F");
        assert_ne!(lie.lie(), F::ZERO, "degree {degree}: the claim is false");
    }
}

#[test]
fn honest_sumchecks_verify() {
    for (index, setting) in LIES.iter().enumerate() {
        for vars in [1, setting.vars, 4] {
            let mut rng = SplitMix64::new(seed(9, index) + vars as u64);
            for trial in 0..20 {
                let mut prover = SmallTranscript::session(seed(9, index), trial);
                let mut verifier = SmallTranscript::session(seed(9, index), trial);
                let verdict =
                    RoundLie::honest(setting.degree, vars, &mut rng, &mut prover, &mut verifier);
                assert!(
                    verdict.accepted,
                    "degree {}, {vars} variables, trial {trial}: an honest proof verifies",
                    setting.degree
                );
            }
        }
    }
}

/// The lie is accepted exactly when `r_0` is a root: scripted at each root, then at values that
/// are not, among them 0, 1 and the elements just below the roots.
#[test]
fn a_lying_round_is_accepted_exactly_at_its_roots() {
    for setting in &LIES {
        for vars in [1, setting.vars, 4] {
            let lie = RoundLie::new(setting.degree, vars);
            let mut rng = SplitMix64::new(0x0104_5C21 + vars as u64);
            let below = FIELD - 1 - setting.degree as u64;
            let others = [0, 1, 2, below, below - 1, 12_345];
            let candidates = lie
                .roots()
                .iter()
                .copied()
                .chain(others.map(F::from_u64_reduced));
            for r in candidates {
                let script = [("stage/0/r/0", r)];
                let verdict = lie.trial(
                    &mut rng,
                    &mut Scripted::new(&script, 7),
                    &mut Scripted::new(&script, 7),
                );
                assert_eq!(verdict.challenge, r, "the scripted challenge");
                assert_eq!(
                    verdict.accepted,
                    lie.roots().contains(&r),
                    "degree {}, {vars} variables, r_0 = {r:?}",
                    setting.degree
                );
            }
        }
    }
}

#[test]
fn honest_lookups_verify_unless_alpha_hits_a_row() {
    for (lookups, rows) in [(1, 1), (5, 3), (16, 16), (3, 17)] {
        let mut rng = SplitMix64::new(0x0104_A11C + rows as u64);
        let honest = Lookups::honest(lookups, rows, &mut rng);
        for trial in 0..20 {
            let mut prover = SmallTranscript::session(seed(8, rows), trial);
            let mut verifier = SmallTranscript::session(seed(8, rows), trial);
            let verdict = honest.trial(&mut prover, &mut verifier);
            assert_eq!(
                verdict.accepted,
                !honest.poles().contains(&verdict.challenge),
                "{lookups} lookups into {rows} rows, trial {trial}"
            );
        }
        // At a row, a leaf's denominator is zero: rejected (RFC-0014 'Soundness budget').
        let row = honest.poles()[0];
        let script = [("logup/alpha", row)];
        let verdict = honest.trial(
            &mut Scripted::new(&script, 3),
            &mut Scripted::new(&script, 3),
        );
        assert!(!verdict.accepted, "α at a row is rejected");
    }
}

/// Small instances of every shape the measurement uses: one lookup into a table tree of height
/// 0, lookup trees with padding, and table trees with and without it.
const SMALL: [(usize, usize); 5] = [(1, 1), (1, 4), (2, 6), (3, 7), (3, 9)];

#[test]
fn bad_lookups_balance_at_lookups_plus_rows_minus_one_values_of_alpha() {
    for (lookups, rows) in SMALL {
        let bad = Lookups::adversarial(lookups, rows);
        assert_eq!(
            bad.accepting().len(),
            lookups + rows - 1,
            "{lookups} lookups outside {rows} rows"
        );
        assert!(
            bad.accepting().iter().all(|a| !bad.poles().contains(a)),
            "no root is a pole"
        );
        assert!(
            bad.keys().iter().all(|k| !bad.rows().contains(k)),
            "every lookup is outside the table"
        );
    }
}

/// The bad lookups are accepted exactly at the identity's roots: scripted at each root, at each
/// pole, and at values that are neither.
#[test]
fn bad_lookups_are_accepted_exactly_at_their_roots() {
    for (lookups, rows) in SMALL {
        let bad = Lookups::adversarial(lookups, rows);
        let others = [2, 1000, FIELD - 1 - (lookups + rows) as u64, 54_321];
        let candidates = bad
            .accepting()
            .iter()
            .chain(bad.poles())
            .copied()
            .chain(others.map(F::from_u64_reduced))
            .collect::<Vec<_>>();
        for alpha in candidates {
            let script = [("logup/alpha", alpha)];
            let verdict = bad.trial(
                &mut Scripted::new(&script, 5),
                &mut Scripted::new(&script, 5),
            );
            assert_eq!(verdict.challenge, alpha, "the scripted challenge");
            assert_eq!(
                verdict.accepted,
                bad.accepting().contains(&alpha),
                "{lookups} lookups outside {rows} rows, α = {alpha:?}"
            );
        }
    }
}

/// RFC-0072 'Testing Strategy' 'Calibration': a non-Boolean bit passes stage 1's booleanity
/// zero-check exactly when the zero-check point zeroes `eq(z, x_0)`. Scripted at each
/// coordinate's root, the others not, and at two roots, it is accepted; at points with no root,
/// among them 0, 1 and the top of the field, it is rejected. Honest bits verify at every point of
/// the real transcript.
#[test]
fn a_non_boolean_bit_passes_booleanity_exactly_when_z_zeroes_its_eq() {
    for vars in [1, 2, 4] {
        let mut rng = SplitMix64::new(0x0526_B001 + vars as u64);
        for trial in 0..20 {
            let honest = NonBoolean::honest(vars, &mut rng);
            let verdict = honest.trial(
                &mut SmallTranscript::session(seed(7, vars), trial),
                &mut SmallTranscript::session(seed(7, vars), trial),
            );
            assert!(
                verdict.accepted,
                "{vars} variables, trial {trial}: honest bits verify"
            );
        }
        let bad = NonBoolean::new(vars, &mut rng);
        let roots = bad.roots();
        // Points: one coordinate at its root, then none, then two.
        let mut points: Vec<Vec<F>> = Vec::new();
        let others = [2, FIELD - 1, 12_345, 7];
        let off = |c: usize| F::from_u64_reduced(others[c % others.len()]);
        for c in 0..vars {
            points.push(
                (0..vars)
                    .map(|j| if j == c { roots[j] } else { off(j) })
                    .collect(),
            );
        }
        points.push((0..vars).map(off).collect());
        // 0 and 1 are a root at each coordinate where the other is not.
        points.push(roots.iter().map(|&r| F::ONE - r).collect());
        if vars >= 2 {
            points.push(
                (0..vars)
                    .map(|j| if j < 2 { roots[j] } else { off(j) })
                    .collect(),
            );
        }
        for z in points {
            let script: Vec<(String, F)> = (0..vars)
                .map(|c| (format!("stage/1/zero/{c}"), z[c]))
                .collect();
            let script: Vec<(&str, F)> = script.iter().map(|(l, x)| (l.as_str(), *x)).collect();
            let verdict = bad.trial(
                &mut Scripted::new(&script, 11),
                &mut Scripted::new(&script, 11),
            );
            assert_eq!(verdict.point, z, "the scripted point");
            let expected = z.iter().zip(&roots).any(|(a, b)| a == b);
            assert_eq!(verdict.accepted, expected, "{vars} variables, z = {z:?}");
            assert_eq!(verdict.accepted, bad.accepts(&z));
        }
    }
}

/// The measurement's own code on the first trials of each setting, a twentieth of them or at
/// most 100,000, on the real transcript: every verdict is the event the analysis predicts, and
/// where ten accepts are expected, at `d = 7` and in every LogUp setting, some happen.
#[test]
fn the_first_trials_of_every_setting_agree_with_the_analysis() {
    let mut runs = Vec::new();
    for (index, setting) in LIES.iter().enumerate() {
        let lie = RoundLie::new(setting.degree, setting.vars);
        let n = (trials(setting.degree) / 20).min(100_000);
        runs.push((n, setting.degree, lie_trials(&lie, seed(1, index), n)));
    }
    for (index, setting) in LOOKUPS.iter().enumerate() {
        let bad = Lookups::adversarial(setting.lookups, setting.rows);
        let roots = bad.accepting().len();
        let n = (trials(roots) / 20).min(100_000);
        runs.push((n, roots, lookup_trials(&bad, seed(2, index), n)));
    }
    for (index, &vars) in ZERO_CHECKS.iter().enumerate() {
        let n = (trials(vars) / 20).min(100_000);
        runs.push((n, vars, booleanity_trials(vars, seed(3, index), n)));
    }
    for (n, roots, accepted) in runs {
        let expected = n as f64 * roots as f64 / FIELD as f64;
        assert!(
            expected < 10.0 || accepted > 0,
            "{roots} roots, {n} trials: {expected:.1} accepts expected, none happened"
        );
    }
}

/// Clopper–Pearson endpoints against an independent oracle, `calibration/oracle.py`: binomial
/// tails summed term by term in mpmath at 50 digits, endpoints bisected to `10^-40`. The last
/// five are reported intervals of the measurement at the scale it runs.
#[test]
fn intervals_match_an_independent_oracle() {
    // (accepted, trials, low, high) at 99%, as the oracle prints them.
    let oracle: [(u64, u64, f64, f64); 14] = [
        (0, 10, 0.0, 0.4112959813475254),
        (1, 10, 0.0005011285754646338, 0.5442870568996868),
        (5, 10, 0.12831055393508325, 0.8716894460649167),
        (10, 10, 0.5887040186524747, 1.0),
        (3, 1000, 0.00033814452906649327, 0.010933777420405242),
        (225, 60_000, 0.0031382972232016906, 0.004441773791599088),
        (
            100,
            6_553_700,
            1.1614908939348302e-05,
            1.9646962416623387e-05,
        ),
        (
            400,
            26_214_800,
            1.3365077295611806e-05,
            1.7335893347505355e-05,
        ),
        (1000, 63_939, 0.014404230426610588, 0.016948262053530453),
        (
            204,
            13_107_400,
            1.2900279437125596e-05,
            1.8596779314959763e-05,
        ),
        (174, 51_201, 0.0027723031631403765, 0.004118553907456106),
        (205, 12_788, 0.013311383367968152, 0.019116599721388582),
        (
            178,
            6_553_700,
            2.2203271716478667e-05,
            3.285730253156574e-05,
        ),
        (
            214,
            3_276_850,
            5.4381165686462664e-05,
            7.770972431378203e-05,
        ),
    ];
    for (k, n, low, high) in oracle {
        let (l, h) = clopper_pearson(k, n, CONFIDENCE);
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs().max(1e-300);
        assert!(close(l, low), "{k} of {n}: low {l:e}, oracle {low:e}");
        assert!(close(h, high), "{k} of {n}: high {h:e}, oracle {high:e}");
    }
}

/// The interval's defining properties, computed by direct summation: each endpoint leaves
/// exactly 0.5% in its tail, and for every rate the intervals that contain it cover at least
/// 99% of the outcomes.
#[test]
fn intervals_leave_half_a_percent_in_each_tail_and_cover_99_percent() {
    let tail = (1.0 - CONFIDENCE) / 2.0;
    // P(X = i) by the definition, C(n, i)·p^i·(1 − p)^{n−i}, and P(X ≤ k) as their sum.
    let pmf = |i: u64, n: u64, p: f64| -> f64 {
        let binomial = (0..i).fold(1.0, |c, j| c * (n - j) as f64 / (j + 1) as f64);
        binomial * p.powi(i as i32) * (1.0 - p).powi((n - i) as i32)
    };
    let cdf = |k: u64, n: u64, p: f64| -> f64 { (0..=k).map(|i| pmf(i, n, p)).sum() };
    for n in [1u64, 2, 7, 30, 120] {
        let intervals: Vec<(f64, f64)> =
            (0..=n).map(|k| clopper_pearson(k, n, CONFIDENCE)).collect();
        for (k, &(low, high)) in (0..=n).zip(&intervals) {
            assert!(low <= k as f64 / n as f64 && k as f64 / n as f64 <= high);
            if k > 0 {
                let upper_tail = 1.0 - cdf(k - 1, n, low);
                assert!(
                    (upper_tail - tail).abs() < 1e-9,
                    "{k} of {n}: P(X ≥ k) at low"
                );
            }
            if k < n {
                assert!(
                    (cdf(k, n, high) - tail).abs() < 1e-9,
                    "{k} of {n}: P(X ≤ k) at high"
                );
                assert!((binomial_cdf(k, n, high) - tail).abs() < 1e-12);
            }
        }
        for step in 1..200 {
            let p = step as f64 / 200.0;
            let coverage: f64 = (0..=n)
                .zip(&intervals)
                .filter(|&(_, &(low, high))| low <= p && p <= high)
                .map(|(k, _)| pmf(k, n, p))
                .sum();
            assert!(
                coverage >= CONFIDENCE - 1e-12,
                "n = {n}, p = {p}: {coverage}"
            );
        }
    }
}

/// `trials` trials of `lie` on the transcript, each checked to be accepted exactly when `r_0` is
/// a root; the accepted count.
fn lie_trials(lie: &RoundLie, seed: u64, trials: u64) -> u64 {
    count_accepted(trials, |trial| {
        let mut rng = SplitMix64::new(seed ^ trial.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let verdict = lie.trial(
            &mut rng,
            &mut SmallTranscript::session(seed, trial),
            &mut SmallTranscript::session(seed, trial),
        );
        assert_eq!(
            verdict.accepted,
            lie.roots().contains(&verdict.challenge),
            "trial {trial}: accepted exactly at a root"
        );
        verdict.accepted
    })
}

/// `trials` trials of `bad` on the transcript, each checked to be accepted exactly when `α` is a
/// root; the accepted count.
fn lookup_trials(bad: &Lookups, seed: u64, trials: u64) -> u64 {
    count_accepted(trials, |trial| {
        let verdict = bad.trial(
            &mut SmallTranscript::session(seed, trial),
            &mut SmallTranscript::session(seed, trial),
        );
        assert_eq!(
            verdict.accepted,
            bad.accepting().binary_search(&verdict.challenge).is_ok(),
            "trial {trial}: accepted exactly at a root"
        );
        verdict.accepted
    })
}

/// `trials` trials of a non-Boolean bit among the bits of a zero-check on `vars` variables, each
/// a fresh table and checked to be accepted exactly when the zero-check point zeroes its `eq`;
/// the accepted count.
fn booleanity_trials(vars: usize, seed: u64, trials: u64) -> u64 {
    count_accepted(trials, |trial| {
        let mut rng = SplitMix64::new(seed ^ trial.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let bad = NonBoolean::new(vars, &mut rng);
        let verdict = bad.trial(
            &mut SmallTranscript::session(seed, trial),
            &mut SmallTranscript::session(seed, trial),
        );
        assert_eq!(
            verdict.accepted,
            bad.accepts(&verdict.point),
            "trial {trial}: accepted exactly when z zeroes eq(z, x_0)"
        );
        verdict.accepted
    })
}

/// One setting's measurement.
struct Row {
    setting: String,
    trials: u64,
    accepted: u64,
    /// The rate the analysis predicts exactly: `K/|F|` for `K` accepting challenges, `δ`'s roots
    /// or the identity's, and `1 − (1 − 1/|F|)^n` for a zero-check point of `n` coordinates.
    exact: f64,
    /// The analytic bound: `d/|F|`, `(N + T)/|F|` or `n/|F|`.
    bound: f64,
    seconds: f64,
}

impl Row {
    fn interval(&self) -> (f64, f64) {
        clopper_pearson(self.accepted, self.trials, CONFIDENCE)
    }

    /// The rate the analysis predicts exactly.
    fn exact(&self) -> f64 {
        self.exact
    }

    /// The analytic bound.
    fn bound(&self) -> f64 {
        self.bound
    }

    /// Whether `K/|F|` is inside the interval: the implementation agrees with the analysis.
    fn inside(&self) -> bool {
        let (low, high) = self.interval();
        low <= self.exact() && self.exact() <= high
    }

    /// Whether the interval's lower end, a one-sided 99.5% lower bound on the rate, is at or below
    /// the analytic bound: the bound holds.
    fn below_bound(&self) -> bool {
        self.interval().0 <= self.bound()
    }
}

/// The measurement (07 §4): every setting's trials, a report on stderr, then the checks: the
/// construction attains the analytic count, `K/|F|` lies in the 99% interval, and the interval's
/// lower end does not exceed the bound.
#[test]
#[ignore = "long: about 3·10^7 trials; the nightly workflow runs it"]
fn cheating_provers_succeed_at_the_analytic_rate() {
    let mut rows = Vec::new();
    for (index, setting) in LIES.iter().enumerate() {
        let start = Instant::now();
        let lie = RoundLie::new(setting.degree, setting.vars);
        let roots = lie.roots().len();
        assert_eq!(
            roots, setting.degree,
            "d roots: the bound d/|F| is attained"
        );
        let trials = trials(roots);
        rows.push(Row {
            setting: format!("sumcheck, d = {}, n = {}", setting.degree, setting.vars),
            trials,
            accepted: lie_trials(&lie, seed(1, index), trials),
            exact: roots as f64 / FIELD as f64,
            bound: setting.degree as f64 / FIELD as f64,
            seconds: start.elapsed().as_secs_f64(),
        });
    }
    for (index, setting) in LOOKUPS.iter().enumerate() {
        let start = Instant::now();
        let bad = Lookups::adversarial(setting.lookups, setting.rows);
        let roots = bad.accepting().len();
        assert_eq!(
            roots,
            setting.lookups + setting.rows - 1,
            "N + T − 1 roots: the most the identity can have"
        );
        let trials = trials(roots);
        rows.push(Row {
            setting: format!("LogUp, N = {}, T = {}", setting.lookups, setting.rows),
            trials,
            accepted: lookup_trials(&bad, seed(2, index), trials),
            exact: roots as f64 / FIELD as f64,
            bound: (setting.lookups + setting.rows) as f64 / FIELD as f64,
            seconds: start.elapsed().as_secs_f64(),
        });
    }
    for (index, &vars) in ZERO_CHECKS.iter().enumerate() {
        let start = Instant::now();
        let trials = trials(vars);
        // `1 − (1 − 1/|F|)^n`: the share of points with a coordinate at its root.
        let miss = 1.0 - 1.0 / FIELD as f64;
        rows.push(Row {
            setting: format!("booleanity, n = {vars}"),
            trials,
            accepted: booleanity_trials(vars, seed(3, index), trials),
            exact: 1.0 - miss.powi(i32::try_from(vars).expect("small")),
            bound: vars as f64 / FIELD as f64,
            seconds: start.elapsed().as_secs_f64(),
        });
    }

    eprintln!(
        "\nSoundness calibration over F65537 (07 §4): trials of cheating provers that the \
         verifier accepted, with 99% Clopper–Pearson intervals. K/|F| is the accepting \
         challenges' share of the field, the exact rate, 1 − (1 − 1/|F|)^n for a zero-check \
         point; bound is d/|F|, (N + T)/|F| or n/|F|.\n"
    );
    eprintln!(
        "{:<24} {:>10} {:>8} {:>8} {:>10} {:>24} {:>10} {:>6} {:>10} {:>6} {:>7}",
        "setting",
        "trials",
        "expected",
        "accepted",
        "rate",
        "99% interval",
        "K/|F|",
        "inside",
        "bound",
        "below",
        "seconds"
    );
    let yes = |b: bool| if b { "yes" } else { "NO" };
    for row in &rows {
        let (low, high) = row.interval();
        eprintln!(
            "{:<24} {:>10} {:>8.1} {:>8} {:>10.4e} {:>24} {:>10.4e} {:>6} {:>10.4e} {:>6} {:>7.1}",
            row.setting,
            row.trials,
            row.trials as f64 * row.exact(),
            row.accepted,
            row.accepted as f64 / row.trials as f64,
            format!("[{low:.4e}, {high:.4e}]"),
            row.exact(),
            yes(row.inside()),
            row.bound(),
            yes(row.below_bound()),
            row.seconds
        );
    }
    let failures: Vec<&str> = rows
        .iter()
        .filter(|row| !(row.inside() && row.below_bound()))
        .map(|row| row.setting.as_str())
        .collect();
    assert!(
        failures.is_empty(),
        "the rate disagrees with the analysis: {}",
        failures.join("; ")
    );
}
