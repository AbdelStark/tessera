// SPDX-License-Identifier: Apache-2.0 OR MIT

//! EXP-0010: the timing probe of secret-indexed multiplicity counting and table reads against
//! the key distribution (README.md in this directory).
//!
//! Issue #257: `tessera_logup::multiplicities` increments `counts[row]` at the row each lookup's
//! key chooses, and a table value is read at a secret key; INV-ARCH-2 says memory access patterns
//! depend only on public data. This harness measures how much the duration of that counting loop
//! depends on the key distribution on one CPU core: seven seeded key streams of the same length
//! are counted into `u32` and `u64` counter arrays of 2^20, 2^24 and 2^28 entries, a value table
//! of the same size is read at the same keys, and a sort-based count and a bitonic sorting network
//! (the oblivious alternative) run on the same streams for context. Every count is checked against
//! an independent sort-based count and, where one exists, a closed form, before and after its arm
//! is timed. Protocol as RFC-0020: 3 warm-ups, 10 measured iterations, median, minimum, maximum
//! and coefficient of variation; above 3% the case is rerun once and flagged if still above.
//!
//! The counting loop is the production loop of `crates/tessera-logup/src/instance.rs` on `main`
//! (this worktree predates that file), reproduced here over both counter widths:
//!
//! ```text
//! let mut counts = vec![0u64; leaves];
//! for (lookup, &row) in lookups.iter().enumerate() {
//!     let row = row as usize;
//!     match counts.get_mut(row) {
//!         Some(count) if row < rows => *count += 1,
//!         _ => return Err(LogUpError::LookupRow { lookup, rows }),
//!     }
//! }
//! ```
//!
//! `MULTLEAK_SMOKE=1` shrinks every size for a quick correctness run; the pre-registered run uses
//! the defaults. Output: a text report on stdout and `results/run.json` next to the manifest.

#![forbid(unsafe_code)]

use std::collections::HashSet;
use std::fmt::Write as _;
use std::hint::black_box;
use std::time::{Duration, Instant};

const WARMUPS: usize = 3;
const ITERATIONS: usize = 10;
const CV_LIMIT: f64 = 0.03;
const HOT_ROWS: usize = 4096;

const SEED_UNIFORM: u64 = 0x7E55_E7A0_0257_0001;
const SEED_ZIPF_RANKS: u64 = 0x7E55_E7A0_0257_0002;
const SEED_ZIPF_HOT: u64 = 0x7E55_E7A0_0257_0003;
const SEED_ALL_EQUAL: u64 = 0x7E55_E7A0_0257_0004;
const SEED_GAUSS_NARROW: u64 = 0x7E55_E7A0_0257_0005;
const SEED_GAUSS_WIDE: u64 = 0x7E55_E7A0_0257_0006;
const SEED_TABLE: u64 = 0x7E55_E7A0_0257_0007;

/// Sizes of one run: the pre-registered defaults, or the smoke test's.
struct Config {
    log_n: u32,
    log_ts: Vec<u32>,
    /// The table size whose streams the sort-based count is timed on (it is the correctness
    /// reference at every size).
    sortcount_log_t: u32,
    /// Bitonic networks: (log2 of the prefix sorted, distributions), on the largest table's streams.
    bitonic: Vec<(u32, Vec<Dist>)>,
}

impl Config {
    fn from_env() -> Self {
        if std::env::var_os("MULTLEAK_SMOKE").is_some() {
            Self {
                log_n: 20,
                log_ts: vec![16, 20],
                sortcount_log_t: 20,
                bitonic: vec![(16, vec![Dist::Uniform, Dist::AllEqual, Dist::GaussNarrow])],
            }
        } else {
            Self {
                log_n: 27,
                log_ts: vec![20, 24, 28],
                sortcount_log_t: 28,
                bitonic: vec![
                    (22, vec![Dist::Uniform, Dist::AllEqual, Dist::GaussNarrow]),
                    (24, vec![Dist::Uniform, Dist::AllEqual]),
                ],
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

    /// Uniform in `[0, 1)` with 53 bits.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// The top `bits` bits of the next output: uniform in `[0, 2^bits)`.
    fn top_bits(&mut self, bits: u32) -> u32 {
        (self.next_u64() >> (64 - bits)) as u32
    }

    /// Two standard normal draws (Box–Muller).
    fn gauss_pair(&mut self) -> (f64, f64) {
        let u1 = 1.0 - self.next_f64(); // (0, 1]
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        (r * theta.cos(), r * theta.sin())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Dist {
    Uniform,
    Zipf,
    Sorted,
    Sequential,
    AllEqual,
    GaussNarrow,
    GaussWide,
}

const DISTS: [Dist; 7] = [
    Dist::Uniform,
    Dist::Zipf,
    Dist::Sorted,
    Dist::Sequential,
    Dist::AllEqual,
    Dist::GaussNarrow,
    Dist::GaussWide,
];

impl Dist {
    fn name(self) -> &'static str {
        match self {
            Dist::Uniform => "uniform",
            Dist::Zipf => "zipf",
            Dist::Sorted => "sorted",
            Dist::Sequential => "sequential",
            Dist::AllEqual => "all_equal",
            Dist::GaussNarrow => "gauss_narrow",
            Dist::GaussWide => "gauss_wide",
        }
    }
}

/// The key stream of `dist`: `n` keys in `[0, 2^log_t)`. `Sorted` needs the `Uniform` stream.
fn stream(dist: Dist, n: usize, log_t: u32, uniform: Option<&[u32]>) -> Vec<u32> {
    let t = 1usize << log_t;
    match dist {
        Dist::Uniform => {
            let mut rng = SplitMix64(SEED_UNIFORM);
            (0..n).map(|_| rng.top_bits(log_t)).collect()
        }
        Dist::Zipf => {
            // A hot set of `HOT_ROWS` distinct rows at uniformly drawn positions; rank
            // `k = ⌊HOT_ROWS^u⌋ + 1`, so `P(rank k) ≈ 1 / ((k − 1) · ln HOT_ROWS)`: Zipf-like,
            // concentrated on a few thousand rows; `hot[k − 1]` is the row.
            let mut hot_rng = SplitMix64(SEED_ZIPF_HOT);
            let mut seen = HashSet::with_capacity(HOT_ROWS);
            let mut hot = Vec::with_capacity(HOT_ROWS);
            while hot.len() < HOT_ROWS {
                let row = hot_rng.top_bits(log_t);
                if seen.insert(row) {
                    hot.push(row);
                }
            }
            let ln_h = (HOT_ROWS as f64).ln();
            let mut rng = SplitMix64(SEED_ZIPF_RANKS);
            (0..n)
                .map(|_| {
                    let u = rng.next_f64();
                    let k = (u * ln_h).exp().floor() as usize + 1;
                    hot[(k - 1).min(HOT_ROWS - 1)]
                })
                .collect()
        }
        Dist::Sorted => {
            let mut keys = uniform.expect("the uniform stream").to_vec();
            keys.sort_unstable();
            keys
        }
        Dist::Sequential => (0..n).map(|i| (i & (t - 1)) as u32).collect(),
        Dist::AllEqual => {
            let row = SplitMix64(SEED_ALL_EQUAL).top_bits(log_t);
            vec![row; n]
        }
        Dist::GaussNarrow | Dist::GaussWide => {
            let (seed, shift) = match dist {
                Dist::GaussNarrow => (SEED_GAUSS_NARROW, 10),
                _ => (SEED_GAUSS_WIDE, 6),
            };
            let sigma = (t >> shift) as f64;
            let centre = (t / 2) as f64;
            let top = (t - 1) as f64;
            let mut rng = SplitMix64(seed);
            let mut keys = Vec::with_capacity(n);
            while keys.len() < n {
                let (z0, z1) = rng.gauss_pair();
                for z in [z0, z1] {
                    if keys.len() < n {
                        keys.push((centre + (sigma * z).round()).clamp(0.0, top) as u32);
                    }
                }
            }
            keys
        }
    }
}

/// The value table read at secret keys: `t` pseudo-random `u32` values.
fn value_table(log_t: u32) -> Vec<u32> {
    let mut rng = SplitMix64(SEED_TABLE);
    (0..1usize << log_t)
        .map(|_| rng.next_u64() as u32)
        .collect()
}

/// The production loop over 4-byte counters (08 §4's counter width). `Err(lookup)` for the first
/// key that is no row.
fn count_u32(keys: &[u32], counts: &mut [u32], rows: usize) -> Result<(), usize> {
    for (lookup, &row) in keys.iter().enumerate() {
        let row = row as usize;
        match counts.get_mut(row) {
            Some(count) if row < rows => *count += 1,
            _ => return Err(lookup),
        }
    }
    Ok(())
}

/// The production loop over 8-byte counters (the production function's width).
fn count_u64(keys: &[u32], counts: &mut [u64], rows: usize) -> Result<(), usize> {
    for (lookup, &row) in keys.iter().enumerate() {
        let row = row as usize;
        match counts.get_mut(row) {
            Some(count) if row < rows => *count += 1,
            _ => return Err(lookup),
        }
    }
    Ok(())
}

/// A table read at every key, accumulated with wrapping addition.
fn gather(keys: &[u32], table: &[u32]) -> u64 {
    let mut acc = 0u64;
    for &k in keys {
        acc = acc.wrapping_add(table[k as usize] as u64);
    }
    acc
}

/// `Σ_j counts[j] · table[j]`, the gather's checksum computed from the counts: an independent
/// check of the table-read arm.
fn gather_from_counts(counts: &[u32], table: &[u32]) -> u64 {
    counts.iter().zip(table).fold(0u64, |acc, (&c, &v)| {
        acc.wrapping_add((c as u64).wrapping_mul(v as u64))
    })
}

/// The sort-based count: copy, sort, run-length into `counts`.
fn sort_count(keys: &[u32], scratch: &mut Vec<u32>, counts: &mut [u32]) {
    scratch.clear();
    scratch.extend_from_slice(keys);
    scratch.sort_unstable();
    run_length_into(scratch, counts);
}

fn run_length_into(sorted: &[u32], counts: &mut [u32]) {
    let mut i = 0;
    while i < sorted.len() {
        let k = sorted[i];
        let mut j = i + 1;
        while j < sorted.len() && sorted[j] == k {
            j += 1;
        }
        counts[k as usize] += (j - i) as u32;
        i = j;
    }
}

/// Batcher's bitonic sorting network, in place, `v.len()` a power of two: `(n/4)·log n·(log n + 1)`
/// compare-exchanges whose addresses depend on the index only. Returns the compare-exchange count.
fn bitonic_sort(v: &mut [u32]) -> u64 {
    let n = v.len();
    let mut exchanges = 0u64;
    let mut k = 2;
    while k <= n {
        let mut j = k / 2;
        while j > 0 {
            for i in 0..n {
                let l = i ^ j;
                if l > i {
                    let ascending = (i & k) == 0;
                    let (a, b) = (v[i], v[l]);
                    let (lo, hi) = (a.min(b), a.max(b));
                    if ascending {
                        v[i] = lo;
                        v[l] = hi;
                    } else {
                        v[i] = hi;
                        v[l] = lo;
                    }
                }
            }
            exchanges += (n / 2) as u64;
            j /= 2;
        }
        k *= 2;
    }
    exchanges
}

/// FNV-1a over the counts, as little-endian `u32`s.
fn digest(counts: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &c in counts {
        for b in c.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

#[derive(Clone, Copy)]
struct Stats {
    median: f64,
    min: f64,
    max: f64,
    cv: f64,
    reran: bool,
    flagged: bool,
}

fn run_once(iteration: &mut dyn FnMut() -> Duration) -> (f64, f64, f64, f64) {
    for _ in 0..WARMUPS {
        iteration();
    }
    let mut s: Vec<f64> = (0..ITERATIONS).map(|_| iteration().as_secs_f64()).collect();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    let median = (s[n / 2 - 1] + s[n / 2]) / 2.0;
    let mean = s.iter().sum::<f64>() / n as f64;
    let var = s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    (median, s[0], s[n - 1], var.sqrt() / mean)
}

/// 3 warm-ups and 10 measured iterations; a coefficient of variation above 3% reruns the case
/// once; the rerun is what is reported, flagged if still above (RFC-0020).
fn measure(mut iteration: impl FnMut() -> Duration) -> Stats {
    let (mut median, mut min, mut max, mut cv) = run_once(&mut iteration);
    let mut reran = false;
    if cv > CV_LIMIT {
        reran = true;
        (median, min, max, cv) = run_once(&mut iteration);
    }
    Stats {
        median,
        min,
        max,
        cv,
        reran,
        flagged: cv > CV_LIMIT,
    }
}

struct Record {
    arm: &'static str,
    log_t: u32,
    dist: Dist,
    keys: usize,
    stats: Stats,
    digest: Option<u64>,
    note: String,
}

fn report(records: &mut Vec<Record>, record: Record) {
    let s = record.stats;
    println!(
        "{:<10} T=2^{:<2} {:<13} n=2^{:<2} median {:>10.3} ms  {:>8.3} ns/key  min {:>10.3}  max {:>10.3}  CV {:>5.2}%{}{}{}",
        record.arm,
        record.log_t,
        record.dist.name(),
        record.keys.trailing_zeros(),
        s.median * 1e3,
        s.median * 1e9 / record.keys as f64,
        s.min * 1e3,
        s.max * 1e3,
        s.cv * 100.0,
        if s.reran { "  (rerun)" } else { "" },
        if s.flagged { "  FLAGGED" } else { "" },
        if record.note.is_empty() {
            String::new()
        } else {
            format!("  {}", record.note)
        },
    );
    records.push(record);
}

fn check(condition: bool, what: &str) {
    assert!(condition, "correctness check failed: {what}");
}

fn main() {
    let config = Config::from_env();
    let n = 1usize << config.log_n;
    println!(
        "EXP-0010 harness: N = 2^{} keys per stream, T ∈ {{{}}}, protocol {} warm-ups, {} iterations, CV limit {:.0}%",
        config.log_n,
        config
            .log_ts
            .iter()
            .map(|t| format!("2^{t}"))
            .collect::<Vec<_>>()
            .join(", "),
        WARMUPS,
        ITERATIONS,
        CV_LIMIT * 100.0
    );
    let t0 = Instant::now();
    let mut records: Vec<Record> = Vec::new();

    for &log_t in &config.log_ts {
        let t = 1usize << log_t;
        let rows = t;
        println!(
            "\n== T = 2^{log_t} ({} MiB of u32 counters) ==",
            (t * 4) >> 20
        );
        let table = value_table(log_t);
        let mut counts32 = vec![0u32; t];
        let mut counts64 = vec![0u64; t];
        let mut counts_ref = vec![0u32; t];
        let mut scratch: Vec<u32> = Vec::with_capacity(n);
        let mut uniform: Option<Vec<u32>> = None;
        let mut uniform_digest = 0u64;
        let mut uniform_gather = 0u64;

        for dist in DISTS {
            let generation = Instant::now();
            let keys = stream(dist, n, log_t, uniform.as_deref());
            let gen_s = generation.elapsed().as_secs_f64();

            // Correctness pass, untimed: direct counts, the production width, the sort-based
            // reference, the closed forms, the gather checksum.
            counts32.fill(0);
            count_u32(&keys, &mut counts32, rows).expect("every key is a row");
            let sum: u64 = counts32.iter().map(|&c| c as u64).sum();
            check(sum == n as u64, "counts sum to N");
            counts64.fill(0);
            count_u64(&keys, &mut counts64, rows).expect("every key is a row");
            check(
                counts32.iter().zip(&counts64).all(|(&a, &b)| a as u64 == b),
                "u32 and u64 counts agree",
            );
            counts_ref.fill(0);
            sort_count(&keys, &mut scratch, &mut counts_ref);
            check(counts32 == counts_ref, "direct and sort-based counts agree");
            match dist {
                Dist::Sequential => {
                    let per_row = (n / t).max(1) as u32;
                    let hit = n.min(t);
                    check(
                        counts32[..hit].iter().all(|&c| c == per_row)
                            && counts32[hit..].iter().all(|&c| c == 0),
                        "sequential closed form",
                    );
                }
                Dist::AllEqual => {
                    let row = keys[0] as usize;
                    check(
                        counts32[row] == n as u32
                            && counts32
                                .iter()
                                .enumerate()
                                .all(|(j, &c)| j == row || c == 0),
                        "all_equal closed form",
                    );
                }
                _ => {}
            }
            let d = digest(&counts32);
            match dist {
                Dist::Uniform => uniform_digest = d,
                Dist::Sorted => check(d == uniform_digest, "sorted reproduces uniform's digest"),
                _ => {}
            }
            let g = gather(&keys, &table);
            check(
                g == gather_from_counts(&counts32, &table),
                "gather checksum equals Σ counts·table",
            );
            match dist {
                Dist::Uniform => uniform_gather = g,
                Dist::Sorted => check(g == uniform_gather, "sorted reproduces uniform's gather"),
                _ => {}
            }
            println!(
                "-- {:<13} generated in {:.2} s; counts digest {:016x}; gather {:016x}; checks passed",
                dist.name(),
                gen_s,
                d,
                g
            );

            // Timed arms. The counter array is zeroed (and so page-resident) before each timed
            // iteration; the timed region is the loop alone.
            let stats = measure(|| {
                counts32.fill(0);
                let start = Instant::now();
                count_u32(&keys, &mut counts32, rows).expect("every key is a row");
                black_box(&counts32);
                start.elapsed()
            });
            check(digest(&counts32) == d, "count32 after timing");
            report(
                &mut records,
                Record {
                    arm: "count32",
                    log_t,
                    dist,
                    keys: n,
                    stats,
                    digest: Some(d),
                    note: String::new(),
                },
            );

            let stats = measure(|| {
                counts64.fill(0);
                let start = Instant::now();
                count_u64(&keys, &mut counts64, rows).expect("every key is a row");
                black_box(&counts64);
                start.elapsed()
            });
            check(
                counts32.iter().zip(&counts64).all(|(&a, &b)| a as u64 == b),
                "count64 after timing",
            );
            report(
                &mut records,
                Record {
                    arm: "count64",
                    log_t,
                    dist,
                    keys: n,
                    stats,
                    digest: Some(d),
                    note: String::new(),
                },
            );

            let stats = measure(|| {
                let start = Instant::now();
                let acc = gather(&keys, &table);
                black_box(acc);
                start.elapsed()
            });
            check(gather(&keys, &table) == g, "gather after timing");
            report(
                &mut records,
                Record {
                    arm: "gather",
                    log_t,
                    dist,
                    keys: n,
                    stats,
                    digest: Some(g),
                    note: String::new(),
                },
            );

            if log_t == config.sortcount_log_t {
                let stats = measure(|| {
                    counts_ref.fill(0);
                    let start = Instant::now();
                    sort_count(&keys, &mut scratch, &mut counts_ref);
                    black_box(&counts_ref);
                    start.elapsed()
                });
                check(counts_ref == counts32, "sort-based count after timing");
                report(
                    &mut records,
                    Record {
                        arm: "sortcount",
                        log_t,
                        dist,
                        keys: n,
                        stats,
                        digest: Some(d),
                        note: "copy + sort_unstable + run-length".to_string(),
                    },
                );
            }

            if Some(&log_t) == config.log_ts.last() {
                for (log_m, dists) in &config.bitonic {
                    if !dists.contains(&dist) {
                        continue;
                    }
                    let m = 1usize << log_m;
                    let prefix = &keys[..m];
                    counts_ref.fill(0);
                    count_u32(prefix, &mut counts_ref, rows).expect("every key is a row");
                    let mut buf = vec![0u32; m];
                    let mut exchanges = 0u64;
                    let stats = measure(|| {
                        let start = Instant::now();
                        buf.copy_from_slice(prefix);
                        exchanges = bitonic_sort(&mut buf);
                        black_box(&buf);
                        start.elapsed()
                    });
                    let expected = (m as u64 / 4) * (*log_m as u64) * (*log_m as u64 + 1);
                    check(exchanges == expected, "bitonic compare-exchange count");
                    check(
                        buf.windows(2).all(|w| w[0] <= w[1]),
                        "bitonic output sorted",
                    );
                    counts32.fill(0);
                    run_length_into(&buf, &mut counts32);
                    check(
                        counts32 == counts_ref,
                        "bitonic run-length count equals direct count",
                    );
                    report(
                        &mut records,
                        Record {
                            arm: "bitonic",
                            log_t,
                            dist,
                            keys: m,
                            stats,
                            digest: None,
                            note: format!(
                                "{} compare-exchanges, {:.3} ns each",
                                exchanges,
                                stats.median * 1e9 / exchanges as f64
                            ),
                        },
                    );
                }
            }

            if dist == Dist::Uniform {
                uniform = Some(keys);
            }
        }
    }

    // Derived quantities per table size, from the medians.
    println!("\n== derived (medians) ==");
    for &log_t in &config.log_ts {
        for arm in ["count32", "count64", "gather"] {
            let find = |d: Dist| {
                records
                    .iter()
                    .find(|r| r.arm == arm && r.log_t == log_t && r.dist == d)
                    .map(|r| r.stats.median)
                    .expect("record")
            };
            let uniform = find(Dist::Uniform);
            let (slowest, t_slow) = DISTS
                .iter()
                .map(|&d| (d, find(d)))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .expect("seven distributions");
            let (fastest, t_fast) = DISTS
                .iter()
                .map(|&d| (d, find(d)))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .expect("seven distributions");
            let typ = (find(Dist::GaussNarrow) + find(Dist::GaussWide)) / 2.0;
            println!(
                "{:<8} T=2^{:<2} R_slow = {:.3} ({}), s = {:.3} (slowest {} {:.1} ms, fastest {} {:.1} ms), g = {:.4}, t_typ/t_uniform = {:.3}",
                arm,
                log_t,
                t_slow / uniform,
                slowest.name(),
                (t_slow - t_fast) / t_slow,
                slowest.name(),
                t_slow * 1e3,
                fastest.name(),
                t_fast * 1e3,
                (find(Dist::GaussWide) - find(Dist::GaussNarrow)).abs() / uniform,
                typ / uniform,
            );
        }
    }

    // JSON for bound.py.
    let mut json = String::new();
    let _ = write!(
        json,
        "{{\"experiment\":\"EXP-0010\",\"log_n\":{},\"warmups\":{},\"iterations\":{},\"cv_limit\":{},\"records\":[",
        config.log_n, WARMUPS, ITERATIONS, CV_LIMIT
    );
    for (i, r) in records.iter().enumerate() {
        let _ = write!(
            json,
            "{}{{\"arm\":\"{}\",\"log_t\":{},\"dist\":\"{}\",\"keys\":{},\"median_s\":{:.9},\"min_s\":{:.9},\"max_s\":{:.9},\"cv\":{:.6},\"reran\":{},\"flagged\":{},\"digest\":{},\"note\":\"{}\"}}",
            if i == 0 { "" } else { "," },
            r.arm,
            r.log_t,
            r.dist.name(),
            r.keys,
            r.stats.median,
            r.stats.min,
            r.stats.max,
            r.stats.cv,
            r.stats.reran,
            r.stats.flagged,
            match r.digest {
                Some(d) => format!("\"{d:016x}\""),
                None => "null".to_string(),
            },
            r.note,
        );
    }
    json.push_str("]}\n");
    let path = format!("{}/results/run.json", env!("CARGO_MANIFEST_DIR"));
    std::fs::write(&path, json).expect("write results/run.json");
    println!("\nwrote {path}; total {:.1} s", t0.elapsed().as_secs_f64());
}
