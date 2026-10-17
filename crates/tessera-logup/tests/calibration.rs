// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Soundness calibration of the LogUp identity over the test field (RFC-0008 'Testing Strategy',
//! 07 §4): the prover runs over `F65537` with challenges in `F65537` itself, where a false accept
//! is frequent enough to measure. `tv-core` checks Goldilocks proofs only, so each trial's roots
//! are checked here as RFC-0008 'Fraction tree' says a verifier does: `d_a ≠ 0`, `d_t ≠ 0` and
//! `n_a·d_t + n_t·d_a = 0`. The prover builds honest trees over the bad leaves, so every later
//! check passes, and the verifier accepts exactly when the roots do; the accepted trials are
//! checked to pass the leaf checks too.
//!
//! An instance with a lookup outside its table balances its roots for at most `N + T − 1` values
//! of `α`, so `(N + T)/|F|` bounds a single random-`α` trial's false-accept rate. Each trial runs
//! the prover on a transcript of its own session, so that `α` is fresh. Two rates are measured,
//! each with its 99% Wilson score interval, whose lower end must stay at or below the bound:
//!
//! - random bad instances, a lookup outside a random table among random lookups: their
//!   difference polynomial has about one root in the field on average, so the rate is near
//!   `1/|F|`, far below the bound;
//! - an adversarial instance, one lookup outside a table of `T` rows with multiplicities chosen
//!   so that the difference polynomial is `c·Π_s (α − z_s)` over `T = N + T − 1` chosen roots
//!   `z_s`: its rate is `T/|F|`, at the bound but for one root.
//!
//! The rates are printed with `--nocapture`.

#![allow(clippy::print_stdout)]

use tessera_codec::Ascii;
use tessera_field::{F65537, Field, PrimeField};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_logup::{LogUpInstance, ProvenLogUp, TreeId, Tuples, prove_logup};
use tessera_sumcheck::{LevelClaim, SumcheckTranscript};

type F = F65537;

/// `|F|`.
const FIELD: f64 = 65537.0;

/// A transcript over `F65537`: messages absorbed as TCE `Vec`, challenges drawn exactly uniformly
/// by rejecting 17-bit words at or above 65537, as `tessera-sumcheck`'s calibration tests draw
/// them.
struct SmallTranscript {
    inner: Transcript,
    words: Vec<u64>,
}

impl SmallTranscript {
    fn new(trial: u64, seed: u8) -> Self {
        let mut digest = [seed; 32];
        digest[..8].copy_from_slice(&trial.to_le_bytes());
        let context = SessionContext {
            protocol: Ascii::new("tessera-protocol/1").expect("ASCII"),
            mode: Mode::NonInteractive,
            params: ParamSetId(Ascii::new("test-small/1").expect("ASCII")),
            statement_digest: digest,
            session_id: [seed; 16],
        };
        Self {
            inner: Transcript::new(&context),
            words: Vec::new(),
        }
    }
}

impl SumcheckTranscript<F> for SmallTranscript {
    fn absorb_message(&mut self, label: &str, message: &[F]) {
        self.inner.absorb(label, &message.to_vec());
    }

    fn challenge(&mut self, label: &str) -> F {
        self.words.clear();
        loop {
            if self.words.is_empty() {
                let out = self.inner.squeeze(label);
                self.words = out
                    .chunks_exact(8)
                    .map(|w| u64::from_le_bytes(w.try_into().expect("8 bytes")) & 0x1_FFFF)
                    .collect();
            }
            if let Some(x) = F::from_canonical_u64(self.words.remove(0)) {
                return x;
            }
        }
    }
}

/// SplitMix64: reproducible inputs.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// A bad instance: the table's rows, the lookups' keys and the committed multiplicities; no
/// values, so that `β` plays no part and the difference polynomial is fixed before `α`.
struct Bad {
    rows: Vec<F>,
    keys: Vec<F>,
    multiplicities: Vec<F>,
}

impl Bad {
    fn prove(&self, trial: u64, seed: u8) -> ProvenLogUp<F> {
        let instance = LogUpInstance::new(
            Tuples::new(&self.rows, None),
            Tuples::new(&self.keys, None),
            &self.multiplicities,
        );
        let mut transcript = SmallTranscript::new(trial, seed);
        prove_logup(&mut transcript, &[instance], &[], &TreeId::paired(1)).expect("well formed")
    }

    /// The verifier's root check, then, on an accepted trial, its leaf checks: the lookup tree's
    /// leaf numerator is 1 and the table tree's leaf denominator is `α − t̃(ρ_t)`.
    fn accepted(&self, proven: &ProvenLogUp<F>) -> bool {
        let [n_a, d_a, n_t, d_t] = proven.roots() else {
            panic!("four roots");
        };
        let roots = *d_a != F::ZERO && *d_t != F::ZERO && *n_a * *d_t + *n_t * *d_a == F::ZERO;
        if roots {
            let leaf = |p: usize| match &proven.leaf_claims()[p] {
                LevelClaim::Fraction(claim) => claim.clone(),
                LevelClaim::Product(_) => panic!("a fraction tree"),
            };
            let (lookup, table) = (leaf(0), leaf(1));
            assert_eq!(
                lookup.numerator,
                F::ONE,
                "the lookup tree's numerators are 1"
            );
            let height = table.point.len();
            let mut padded = self.rows.clone();
            padded.resize(1 << height, self.rows[0]);
            let eval = tessera_poly::evaluate(&padded, &table.point).expect("2^h rows");
            assert_eq!(table.denominator, *proven.alpha() - eval, "α − t̃(ρ_t)");
        }
        roots
    }
}

/// The 99% Wilson score interval of `k` successes in `n` trials.
fn wilson(k: u64, n: u64) -> (f64, f64) {
    const Z: f64 = 2.575_829_303_548_901;
    let (k, n) = (k as f64, n as f64);
    let p = k / n;
    let denominator = 1.0 + Z * Z / n;
    let center = (p + Z * Z / (2.0 * n)) / denominator;
    let half = Z * (p * (1.0 - p) / n + Z * Z / (4.0 * n * n)).sqrt() / denominator;
    ((center - half).max(0.0), (center + half).min(1.0))
}

/// `trials` single random-`α` trials of instances from `instance`, and the 99% interval of their
/// false-accept rate.
fn measure(trials: u64, seed: u8, mut instance: impl FnMut(u64) -> Bad) -> (u64, (f64, f64)) {
    let mut accepted = 0;
    for trial in 0..trials {
        let bad = instance(trial);
        let proven = bad.prove(trial, seed);
        if bad.accepted(&proven) {
            accepted += 1;
        }
    }
    (accepted, wilson(accepted, trials))
}

/// Random bad instances: a table of `T = 15` rows of distinct random keys, 16 lookups of random
/// rows, one replaced by a key outside the table, and multiplicities that a cheating prover
/// commits at random in the field. At each `α` that is no pole, the difference
/// `Σ_i 1/(α − a_i) − Σ_j m_j/(α − t_j)` is then uniform, so the rate is about `1/|F|`, against
/// the bound `(N + T)/|F| = 31/65537`. (With the honest multiplicities of the lookups in the
/// table, the difference is `1/(α − a)` of the one outside, which never vanishes: such an
/// instance is never accepted.) Returns the accepted trials and their 99% interval.
fn random_trials(trials: u64, seed: u64) -> (u64, (f64, f64)) {
    let mut rng = SplitMix64(seed);
    let (rows, lookups) = (15usize, 16usize);
    let instance = |_: u64| {
        // Distinct keys, one more than the rows: the last is outside the table.
        let mut keys: Vec<F> = Vec::with_capacity(rows + 1);
        while keys.len() <= rows {
            let k = F::from_u64_reduced(rng.next_u64());
            if !keys.contains(&k) {
                keys.push(k);
            }
        }
        let outside = keys.pop().expect("a key outside the table");
        let chosen: Vec<usize> = (0..lookups)
            .map(|_| rng.below(rows as u64) as usize)
            .collect();
        let replaced = rng.below(lookups as u64) as usize;
        let m: Vec<F> = (0..1 << 4)
            .map(|_| F::from_u64_reduced(rng.next_u64()))
            .collect();
        let mut lookup_keys: Vec<F> = chosen.iter().map(|&r| keys[r]).collect();
        lookup_keys[replaced] = outside;
        Bad {
            rows: keys,
            keys: lookup_keys,
            multiplicities: m,
        }
    };
    let (accepted, (low, high)) = measure(trials, 1, instance);
    let bound = 31.0 / FIELD;
    println!(
        "random bad instances: {accepted} of {trials} accepted, rate {:.3e}, 99% interval \
         [{low:.3e}, {high:.3e}], bound (N + T)/|F| = {bound:.3e}, 1/|F| = {:.3e}",
        accepted as f64 / trials as f64,
        1.0 / FIELD
    );
    assert!(
        low <= bound,
        "the interval's lower end is at or below the bound"
    );
    assert!(
        high < bound / 4.0,
        "random instances accept far below the bound"
    );
    (accepted, (low, high))
}

#[test]
fn random_bad_instances_accept_far_below_the_bound() {
    let _ = random_trials(200_000, 0x60_3001);
}

/// The same measurement over enough trials to see accepts, which the nightly workflow runs:
/// `cargo test --locked -p tessera-logup --test calibration -- --ignored --nocapture`.
#[test]
#[ignore = "long: 4·10^6 trials, about a minute; the nightly workflow runs it"]
fn random_bad_instances_accept_far_below_the_bound_over_millions_of_trials() {
    let (accepted, _) = random_trials(4_000_000, 0x60_3003);
    assert!(accepted > 0, "enough trials to see accepts");
}

/// The adversarial instance: `N = 1` lookup of the key `T`, outside the table of keys
/// `0, …, T − 1`, and multiplicities making the difference polynomial
/// `R(α) = 1/(α − a) − Σ_j m_j/(α − j)` equal to `c·Π_s (α − z_s)/((α − a)·Π_j (α − j))`, with the
/// `T` roots `z_s = T + 1 + s`: the residue at `a` must be 1, which gives `c`, and the residue at
/// `j` is `−m_j`. The identity holds for exactly the `T = N + T − 1` values `z_s` of `α`, none a
/// pole, so the rate is `T/|F|` against the bound `(T + 1)/|F|`.
#[test]
fn an_adversarial_instance_accepts_at_the_bound() {
    let rows = 255usize;
    let trials = 60_000;
    let t = |j: usize| F::from_u64_reduced(j as u64);
    let a = t(rows);
    let roots: Vec<F> = (0..rows).map(|s| t(rows + 1 + s)).collect();
    let product = |x: F, of: &mut dyn Iterator<Item = F>| of.fold(F::ONE, |p, y| p * (x - y));
    let c = product(a, &mut (0..rows).map(t))
        * product(a, &mut roots.iter().copied())
            .inverse()
            .expect("no root at a");
    let mut m = vec![F::ZERO; 1 << 8];
    for (j, m) in m.iter_mut().enumerate().take(rows) {
        let numerator = c * product(t(j), &mut roots.iter().copied());
        let denominator = (t(j) - a) * product(t(j), &mut (0..rows).filter(|&l| l != j).map(t));
        *m = -(numerator * denominator.inverse().expect("distinct poles"));
    }
    // The identity at each root, and not elsewhere: R(z_s) = 0, R(z) ≠ 0 at a few other points.
    let r = |x: F| {
        (x - a).inverse().expect("not a pole")
            - (0..rows).fold(F::ZERO, |s, j| {
                s + m[j] * (x - t(j)).inverse().expect("not a pole")
            })
    };
    assert!(
        roots.iter().all(|&z| r(z) == F::ZERO),
        "every chosen root balances"
    );
    assert!(
        (0..20).all(|k| r(t(2 * rows + 5 + k)) != F::ZERO),
        "nothing else does"
    );

    let bad = Bad {
        rows: (0..rows).map(t).collect(),
        keys: vec![a],
        multiplicities: m,
    };
    let (accepted, (low, high)) = measure(trials, 2, |_| Bad {
        rows: bad.rows.clone(),
        keys: bad.keys.clone(),
        multiplicities: bad.multiplicities.clone(),
    });
    let bound = (rows + 1) as f64 / FIELD;
    let expected = rows as f64 / FIELD;
    println!(
        "adversarial instance: {accepted} of {trials} accepted, rate {:.3e}, 99% interval \
         [{low:.3e}, {high:.3e}], bound (N + T)/|F| = {bound:.3e}, its {rows} roots give \
         {expected:.3e}",
        accepted as f64 / trials as f64
    );
    assert!(
        low <= bound,
        "the interval's lower end is at or below the bound"
    );
    assert!(
        low <= expected && expected <= high,
        "the rate is the roots' share of the field"
    );
}
