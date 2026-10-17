// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Exact binomial confidence intervals (Clopper and Pearson, 1934), with no dependency.
//!
//! For `k` successes in `n` trials, the two-sided interval at confidence `1 − a` is `[low, high]`
//! with `P(X ≥ k) = a/2` at `p = low` and `P(X ≤ k) = a/2` at `p = high`, `X ~ Binomial(n, p)`;
//! `low = 0` when `k = 0` and `high = 1` when `k = n`. It inverts the two one-sided binomial
//! tests, so it contains a rate `p` exactly when neither rejects `p` at level `a/2`, and it covers
//! every `p` with probability at least `1 − a`.

/// `P(X ≤ k)` for `X ~ Binomial(n, p)`.
///
/// The terms are summed in logarithms, from `ln P(X = 0) = n·ln(1 − p)` by the ratio
/// `P(X = i + 1)/P(X = i) = (n − i)/(i + 1) · p/(1 − p)`, so that trials by the million at rates
/// near `10^-5` neither underflow nor lose the tail. The `k + 1` terms' relative errors grow
/// with `k`, to about `k·10^-15`.
pub fn binomial_cdf(k: u64, n: u64, p: f64) -> f64 {
    if k >= n || p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return 0.0;
    }
    let (ln_p, ln_q) = (p.ln(), (-p).ln_1p());
    let mut term = n as f64 * ln_q;
    // A running log-sum-exp: the largest term so far and the sum of exp(term − largest).
    let (mut largest, mut sum) = (term, 1.0f64);
    for i in 0..k {
        term += ((n - i) as f64).ln() - ((i + 1) as f64).ln() + ln_p - ln_q;
        if term > largest {
            sum = sum * (largest - term).exp() + 1.0;
            largest = term;
        } else {
            sum += (term - largest).exp();
        }
    }
    (largest + sum.ln()).exp().min(1.0)
}

/// The root in `(0, 1)` of a function that decreases from positive to negative, by bisection
/// until the bracket's ends are adjacent doubles.
fn root(f: impl Fn(f64) -> f64) -> f64 {
    let (mut low, mut high) = (0.0f64, 1.0f64);
    loop {
        let mid = 0.5 * (low + high);
        if mid <= low || mid >= high {
            return mid;
        }
        if f(mid) > 0.0 {
            low = mid;
        } else {
            high = mid;
        }
    }
}

/// The two-sided Clopper–Pearson interval `(low, high)` at `confidence` of the rate of
/// `successes` in `trials`.
pub fn clopper_pearson(successes: u64, trials: u64, confidence: f64) -> (f64, f64) {
    assert!(
        successes <= trials && trials > 0,
        "{successes} successes in {trials} trials"
    );
    let tail = (1.0 - confidence) / 2.0;
    let low = match successes {
        0 => 0.0,
        // P(X ≥ k) = 1 − P(X ≤ k − 1) rises with p.
        k => root(|p| binomial_cdf(k - 1, trials, p) - (1.0 - tail)),
    };
    let high = match successes == trials {
        true => 1.0,
        false => root(|p| binomial_cdf(successes, trials, p) - tail),
    };
    (low, high)
}
