// SPDX-License-Identifier: Apache-2.0 OR MIT
//! EXP-0035 (EXP-0033's harness, one configuration): the rate at which an epoch's revealed blinding coefficients lose rank under the
//! tensor proximity generator (G-T), the same under R1 (G-T1), and uniform coefficients (G-U),
//! over `Fp2 = Fp[X]/(X^2 + 1)` for small primes `p ≡ 3 (mod 4)`. See README.md.
#![forbid(unsafe_code)]

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

const SEED: u64 = 0x0035;
const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
const TRIALS: u64 = 100_000;
const PRIMES: [u64; 1] = [7];
const SESSIONS: [usize; 4] = [8, 16, 32, 64];
const OFFSETS: [usize; 1] = [3];

struct SplitMix(u64);
impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(GOLDEN);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[0, p)` by rejection.
    fn below(&mut self, p: u64) -> u64 {
        let zone = u64::MAX - u64::MAX % p;
        loop {
            let x = self.next();
            if x < zone {
                return x % p;
            }
        }
    }
}

/// `a + bX` with `X^2 = −1`.
#[derive(Clone, Copy)]
struct F2(u64, u64);
fn mul(p: u64, x: F2, y: F2) -> F2 {
    F2((x.0 * y.0 + (p - x.1) * y.1) % p, (x.0 * y.1 + x.1 * y.0) % p)
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Gen {
    T,
    #[allow(dead_code)] // kept from EXP-0033's harness; this grid runs G-T only
    T1,
    U,
    /// Control 2: G-T at a point with every coordinate in `Fp`.
    TReal,
}

/// `eq(ξ, idx)` over `ξ.len()` bits, bit `t` of `idx` against coordinate `t`.
fn eq(p: u64, xi: &[F2], idx: usize) -> F2 {
    xi.iter().enumerate().fold(F2(1, 0), |acc, (t, &x)| {
        let f = if idx >> t & 1 == 1 { x } else { F2((1 + p - x.0) % p, (p - x.1) % p) };
        mul(p, acc, f)
    })
}

/// Fp-rank by Gaussian elimination; `m` is row-major with `cols` columns.
fn rank(p: u64, m: &mut [u64], rows: usize, cols: usize) -> usize {
    let mut r = 0;
    for c in 0..cols {
        let Some(piv) = (r..rows).find(|&i| m[i * cols + c] != 0) else { continue };
        for j in 0..cols {
            m.swap(r * cols + j, piv * cols + j);
        }
        let inv = pow(p, m[r * cols + c], p - 2);
        for j in c..cols {
            m[r * cols + j] = m[r * cols + j] * inv % p;
        }
        for i in 0..rows {
            let f = m[i * cols + c];
            if i != r && f != 0 {
                for j in c..cols {
                    m[i * cols + j] = (m[i * cols + j] + (p - f) * m[r * cols + j]) % p;
                }
            }
        }
        r += 1;
        if r == rows {
            break;
        }
    }
    r
}

fn pow(p: u64, mut b: u64, mut e: u64) -> u64 {
    let mut acc = 1;
    while e > 0 {
        if e & 1 == 1 {
            acc = acc * b % p;
        }
        b = b * b % p;
        e >>= 1;
    }
    acc
}

/// One trial: is the `4N × B` blinding-coefficient matrix rank-deficient (event E)?
fn trial(p: u64, g: Gen, n: usize, dn: usize, rng: &mut SplitMix, m: &mut Vec<u64>) -> bool {
    let b = 6 * n + 8;
    let bits = usize::BITS as usize - (dn + b - 1).leading_zeros() as usize;
    let rows = 4 * n;
    m.clear();
    m.resize(rows * b, 0);
    let mut put = |row: usize, col: usize, c: F2| {
        m[row * b + col] = c.0;
        m[(row + 1) * b + col] = c.1;
    };
    let mut xi = vec![F2(0, 0); bits];
    for i in 0..n {
        let (beta, f) = (4 + 6 * i, 8 + 6 * i);
        // Evaluation row: (1, X) on β_i.
        put(4 * i, beta, F2(1, 0));
        put(4 * i, beta + 1, F2(0, 1));
        // Proximity row.
        for x in xi.iter_mut() {
            *x = match g {
                Gen::TReal => F2(rng.below(p), 0),
                _ => F2(rng.below(p), rng.below(p)),
            };
        }
        for r in 0..b {
            let c = match g {
                Gen::T | Gen::TReal => eq(p, &xi, dn + r),
                Gen::T1 | Gen::U if r == f => F2(1, 0),
                Gen::T1 | Gen::U if r == f + 1 => F2(0, 1),
                Gen::T1 => eq(p, &xi, dn + r),
                Gen::U => F2(rng.below(p), rng.below(p)),
            };
            put(4 * i + 2, r, c);
        }
    }
    rank(p, m, rows, b) < rows
}

/// Control 1: rank against a brute-force left-null-space count at p = 3.
fn control_rank() -> bool {
    let p = 3;
    let mut rng = SplitMix(SEED ^ 0xC1);
    (0..10_000).all(|_| {
        let rows = 1 + rng.below(4) as usize;
        let cols = 1 + rng.below(6) as usize;
        let m: Vec<u64> = (0..rows * cols).map(|_| rng.below(p)).collect();
        let mut null = 0u64;
        for code in 0..p.pow(rows as u32) {
            let x: Vec<u64> = (0..rows).map(|i| code / p.pow(i as u32) % p).collect();
            if (0..cols).all(|c| (0..rows).map(|i| x[i] * m[i * cols + c]).sum::<u64>() % p == 0) {
                null += 1;
            }
        }
        let brute = rows - (null.ilog(p) as usize);
        rank(p, &mut m.clone(), rows, cols) == brute
    })
}

fn rate(p: u64, g: Gen, n: usize, dn: usize, trials: u64, seed: u64) -> u64 {
    let mut rng = SplitMix(seed);
    let mut m = Vec::new();
    (0..trials).filter(|_| trial(p, g, n, dn, &mut rng, &mut m)).count() as u64
}

fn main() {
    let start = std::time::Instant::now();
    let c1 = control_rank();
    let c2 = rate(31, Gen::TReal, 2, 0, 10_000, SEED ^ 0xC2);
    let c3 = rate(127, Gen::U, 2, 0, 10_000, SEED ^ 0xC3);
    println!("control 1 (rank = brute force, 10,000 matrices at p = 3): {}", if c1 { "pass" } else { "FAIL" });
    println!("control 2 (G-T at real points, p = 31, N = 2): {c2}/10000 in E: {}", if c2 == 10_000 { "pass" } else { "FAIL" });
    println!("control 3 (G-U, p = 127, N = 2): {c3}/10000 in E: {}", if c3 <= 10 { "pass" } else { "FAIL" });

    let mut cells = Vec::new();
    for g in [Gen::T] {
        for n in SESSIONS {
            for dn in OFFSETS {
                for p in PRIMES {
                    cells.push((g, n, dn, p));
                }
            }
        }
    }
    let next = AtomicUsize::new(0);
    let out = Mutex::new(vec![0u64; cells.len()]);
    std::thread::scope(|s| {
        for _ in 0..std::thread::available_parallelism().map_or(8, |n| n.get()) {
            s.spawn(|| {
                loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&(g, n, dn, p)) = cells.get(k) else { break };
                    let seed = SEED.wrapping_add((k as u64).wrapping_mul(GOLDEN));
                    let f = rate(p, g, n, dn, TRIALS, seed);
                    out.lock().unwrap()[k] = f;
                    eprintln!("cell {k}: {g:?} N={n} Dn={dn} p={p}: {f}");
                }
            });
        }
    });
    let out = out.into_inner().unwrap();

    println!("\ngenerator N Dn p trials failures");
    for (k, &(g, n, dn, p)) in cells.iter().enumerate() {
        println!("{g:?} {n} {dn} {p} {TRIALS} {}", out[k]);
    }
    eprintln!("wall {:.1} s", start.elapsed().as_secs_f64());
}
