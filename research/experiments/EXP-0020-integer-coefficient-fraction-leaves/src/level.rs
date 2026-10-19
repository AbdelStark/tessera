// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A tree level's sumcheck without an `eq` table (RFC-0010 'Tree levels'), sending the messages
//! the batch prover would.
//!
//! A level proves, for each instance `i` on `n_i` variables with claim `c_i` at `r_i`,
//! `Σ_x eq(r_i, x)·P_i(x) = c_i`. `P_i` is a product forest's `Ṽ(0 ‖ x)·Ṽ(1 ‖ x)`, or a
//! fraction tree's `N(0 ‖ x)·D(1 ‖ x) + N(1 ‖ x)·D(0 ‖ x) + λ·D(0 ‖ x)·D(1 ‖ x)`. The batch
//! prover would take `eq(r_i, ·)` as a public table: built for the level, bound with the others
//! after every round, three node sums per round. This prover sends the same messages, so the same
//! proof, from less work (Dao and Thaler, ePrint 2024/1210; Gruen, ePrint 2024/108, §3.2):
//!
//! - **`eq` stays split.** In own round `k`, with the instance's variables below `k` bound to
//!   `c_0, …, c_{k−1}`, `eq(r_i, ·)` is `s·ℓ(X)·eq(r_rest, ·)`:
//!   - `s = Π_{t<k} ((1 − r_t)(1 − c_t) + r_t·c_t)`;
//!   - `ℓ(X) = (1 − r_k)(1 − X) + r_k·X`;
//!   - `eq(r_rest, ·)`, over the free variables, is the product of `eq_lo` over their lower half
//!     and `eq_hi` over their upper half, each about `√N` entries, rebuilt each round and never
//!     bound.
//! - **Two node sums.** `v(z) = Σ_x eq(r_rest, x)·P(z ‖ x)` has degree 2, and the round
//!   polynomial is `p(z) = s·ℓ(z)·v(z)`. The prover sums `v(0)` and `v(2)` over the pairs.
//!   `c = p(0) + p(1)` gives `v(1) = (c − s·ℓ(0)·v(0)) / (s·ℓ(1))`, and then
//!   `v(3) = v(0) + 3·(v(2) − v(1))`. When `s·ℓ(1) = s·r_k = 0`, an event of the public
//!   challenges of probability about `2^-128` per round, it sums `v(1)` as well.
//! - **Claims.** After the challenge `c`, the instance's claim is `p(c)`, from `v(0)`, `v(1)` and
//!   `v(2)`, and `s` becomes `s·ℓ(c)`. The caller gives the level's claims: a tree level has
//!   them, from the root values or from the level above.
//!
//! A padding round adds `γ^i·2^{n_max − n_i − j − 1}·c_i` at every node, as the batch prover does
//! with `c_i` summed from the tables. The first own round reads the tables unbound, in the prime
//! subfield where they are, and binds them; later rounds bind in place.
//!
//! Control flow depends on shapes and on public challenges only, never on a table's value
//! (INV-ARCH-2). The bound tables are [`SecretBuf`]s, and each instance's node sums and claim
//! clear themselves when it drops (05 §3 OBS-6).

use core::ops::{Add, Mul, Sub};

use tessera_codec::{SecretBuf, Zeroize};
use tessera_field::Field;
use tessera_poly::PolyField;

use crate::contract::eq_weights;
use crate::kernel::{bind_base, bind_ext, bind_in_place};
use crate::prover::{Labels, padding_scale};
use crate::{Prefix, ProvenBatch, SumcheckProof, SumcheckTranscript};

/// A fraction tree's numerators on the level below the one proven, each half its own table.
pub(crate) enum LevelNumerators<'a, F: crate::BenchField> {
    /// Leaves whose numerators are all 1, as a LogUp lookup tree's: no table.
    Unit,
    /// In the prime subfield, as a LogUp table tree's leaves `−m_j`.
    Base {
        even: &'a [F::Base],
        odd: &'a [F::Base],
    },
    /// In `F`, every level above the leaves.
    Ext { even: &'a [F], odd: &'a [F] },
}

/// The tables of the level below the one proven: `V(0 ‖ ·)` and `V(1 ‖ ·)`, its even and odd
/// halves.
pub(crate) enum LevelTables<'a, F: crate::BenchField> {
    /// A product forest's level, in the prime subfield.
    Product {
        even: &'a [F::Base],
        odd: &'a [F::Base],
    },
    /// A fraction tree's level, with the level's `λ`.
    Fraction {
        numerators: LevelNumerators<'a, F>,
        d_even: &'a [F],
        d_odd: &'a [F],
        lambda: F,
        raw: Option<&'a crate::coeff::Raw<F>>,
    },
}

/// One instance of a tree level: its point `r_i`, lowest variable first; its claim `c_i`, the
/// sum it proves; and the level below.
pub(crate) struct LevelInstance<'a, F: crate::BenchField> {
    pub(crate) point: &'a [F],
    pub(crate) claim: F,
    pub(crate) tables: LevelTables<'a, F>,
}

/// The values of a table on the line through a pair `(a, b)` at the nodes `0, 1, 2`: `a`, `b` and
/// `b + (b − a)`.
#[inline(always)]
fn line<E: Copy + Add<Output = E> + Sub<Output = E>>(a: E, b: E) -> [E; 3] {
    [a, b, b + (b - a)]
}

/// `Σ_hi eq_hi[hi]·Σ_lo eq_lo[lo]·P(pair)` at the nodes `0, 2`, and `1` when `ONE`, over the pairs
/// of a product's two tables, in their field `E`; `lift(w, x)` is `w·x` for a weight `w ∈ F`.
fn product_sums<F, E, const ONE: bool>(
    even: &[E],
    odd: &[E],
    eq_lo: &[F],
    eq_hi: &[F],
    lift: impl Fn(F, E) -> F,
) -> [F; 3]
where
    F: Field,
    E: Copy + Add<Output = E> + Sub<Output = E> + Mul<Output = E>,
{
    let block = 2 * eq_lo.len();
    let mut sums = [F::ZERO; 3];
    for ((even, odd), &w_hi) in even
        .chunks_exact(block)
        .zip(odd.chunks_exact(block))
        .zip(eq_hi)
    {
        let mut inner = [F::ZERO; 3];
        for ((e, o), &w_lo) in even.chunks_exact(2).zip(odd.chunks_exact(2)).zip(eq_lo) {
            let (e, o) = (line(e[0], e[1]), line(o[0], o[1]));
            inner[0] += lift(w_lo, e[0] * o[0]);
            if ONE {
                inner[1] += lift(w_lo, e[1] * o[1]);
            }
            inner[2] += lift(w_lo, e[2] * o[2]);
        }
        for (sum, part) in sums.iter_mut().zip(inner) {
            *sum += w_hi * part;
        }
    }
    sums
}

/// A fraction tree's `N(0)·D(1) + N(1)·D(0) + λ·D(0)·D(1)` at one node, from its numerators'
/// values `n` and its denominators' `d`, with `num(d, n) = d·n`.
#[inline(always)]
fn fraction_at<F: Field, N: Copy>(n: [N; 2], d: [F; 2], lambda: F, num: impl Fn(F, N) -> F) -> F {
    num(d[1], n[0]) + num(d[0], n[1]) + lambda * d[0] * d[1]
}

/// As [`product_sums`] for a fraction tree, whose numerators are in `N` (`num(d, n) = d·n`), or
/// all 1 when `numerators` is `None`.
fn fraction_sums<F, N, const ONE: bool>(
    numerators: Option<(&[N], &[N])>,
    d_even: &[F],
    d_odd: &[F],
    lambda: F,
    eq_lo: &[F],
    eq_hi: &[F],
    num: impl Fn(F, N) -> F,
) -> [F; 3]
where
    F: crate::BenchField,
    N: Copy + Add<Output = N> + Sub<Output = N>,
{
    if crate::arm() != 0 {
        return crate::arith::fraction::<F, N, ONE>(
            numerators, d_even, d_odd, lambda, eq_lo, eq_hi, num,
        );
    }
    let block = 2 * eq_lo.len();
    let mut sums = [F::ZERO; 3];
    let rows = d_even.chunks_exact(block).zip(d_odd.chunks_exact(block));
    match numerators {
        None => {
            for ((d_even, d_odd), &w_hi) in rows.zip(eq_hi) {
                let mut inner = [F::ZERO; 3];
                for ((d0, d1), &w_lo) in
                    d_even.chunks_exact(2).zip(d_odd.chunks_exact(2)).zip(eq_lo)
                {
                    let (d0, d1) = (line(d0[0], d0[1]), line(d1[0], d1[1]));
                    for z in [0, 1, 2] {
                        if z != 1 || ONE {
                            // N = 1: D(1) + D(0) + λ·D(0)·D(1).
                            inner[z] += w_lo * (d1[z] + d0[z] + lambda * d0[z] * d1[z]);
                        }
                    }
                }
                for (sum, part) in sums.iter_mut().zip(inner) {
                    *sum += w_hi * part;
                }
            }
        }
        Some((n_even, n_odd)) => {
            let numerators = n_even.chunks_exact(block).zip(n_odd.chunks_exact(block));
            for (((n_even, n_odd), (d_even, d_odd)), &w_hi) in numerators.zip(rows).zip(eq_hi) {
                let mut inner = [F::ZERO; 3];
                let pairs = n_even
                    .chunks_exact(2)
                    .zip(n_odd.chunks_exact(2))
                    .zip(d_even.chunks_exact(2).zip(d_odd.chunks_exact(2)));
                for (((n0, n1), (d0, d1)), &w_lo) in pairs.zip(eq_lo) {
                    let (n0, n1) = (line(n0[0], n0[1]), line(n1[0], n1[1]));
                    let (d0, d1) = (line(d0[0], d0[1]), line(d1[0], d1[1]));
                    for z in [0, 1, 2] {
                        if z != 1 || ONE {
                            let value = fraction_at([n0[z], n1[z]], [d0[z], d1[z]], lambda, &num);
                            inner[z] += w_lo * value;
                        }
                    }
                }
                for (sum, part) in sums.iter_mut().zip(inner) {
                    *sum += w_hi * part;
                }
            }
        }
    }
    sums
}

/// A table's values lifted from the prime subfield.
fn lift_base<F: crate::BenchField>(values: &[F::Base]) -> impl Iterator<Item = F> + '_ {
    values.iter().map(|&b| F::ZERO.add_base(b))
}

/// `Σ_x eq(point, x)·even[x]·odd[x]`: a product forest's root claim at its root point, the sum
/// its first level proves, from the halves of its level 1, with `eq` split as the rounds split it.
pub(crate) fn product_claim<F: crate::BenchField>(
    point: &[F],
    even: &[F::Base],
    odd: &[F::Base],
) -> F {
    let (lo, hi) = point.split_at(point.len().div_ceil(2));
    let (eq_lo, eq_hi) = (eq_weights(lo), eq_weights(hi));
    let block = eq_lo.len();
    let mut sum = F::ZERO;
    for ((even, odd), &w_hi) in even
        .chunks_exact(block)
        .zip(odd.chunks_exact(block))
        .zip(&eq_hi)
    {
        let mut inner = F::ZERO;
        for ((&e, &o), &w_lo) in even.iter().zip(odd).zip(&eq_lo) {
            inner += w_lo.mul_by_base(e * o);
        }
        sum += w_hi * inner;
    }
    sum
}

/// One instance while the level's rounds run.
struct Split<'a, F: crate::BenchField> {
    tables: &'a LevelTables<'a, F>,
    point: &'a [F],
    /// `n_max − n_i`: the padding rounds, which the batch runs first.
    pad: usize,
    /// `γ^i`.
    weight: F,
    /// The sum the instance's remaining rounds prove.
    claim: F,
    /// `s`, the product of `ℓ_t(c_t)` over the instance's bound variables.
    scale: F,
    /// `v(0), v(1), v(2)` of the current own round, for the claim after its challenge.
    sums: [F; 3],
    /// The tables as the rounds bind them, in the order of the final evaluations: `[V(0), V(1)]`
    /// for a product, `[N(0), N(1), D(0), D(1)]` for a fraction tree, the numerators empty when
    /// they are 1. Empty until the first own round binds them.
    bound: Vec<SecretBuf<F>>,
    window: Option<crate::coeff::Window<F>>,
}

/// The bound tables clear themselves; the node sums and the claim, which no message carries alone,
/// are cleared here (05 §3 OBS-6).
impl<F: crate::BenchField> Drop for Split<'_, F> {
    fn drop(&mut self) {
        self.sums.zeroize();
        self.claim.zeroize();
    }
}

impl<'a, F: crate::BenchField> Split<'a, F> {
    fn vars(&self) -> usize {
        self.point.len()
    }

    /// Adds `γ^i` times the instance's round polynomial at `0, 2, 3` into `message`, for round
    /// `j` counted from 0.
    fn add_round(&mut self, j: usize, message: &mut [F; 3]) {
        if j < self.pad {
            let value = self.weight * self.claim.mul_by_base(padding_scale(self.pad - 1 - j));
            for entry in message.iter_mut() {
                *entry += value;
            }
            return;
        }
        let k = j - self.pad;
        let r = self.point[k];
        let rest = &self.point[k + 1..];
        let (lo, hi) = rest.split_at(rest.len().div_ceil(2));
        let (eq_lo, eq_hi) = (eq_weights(lo), eq_weights(hi));
        // s·ℓ(0) and s·ℓ(1); s·ℓ(z) = (1 − z)·s·ℓ(0) + z·s·ℓ(1).
        let at_zero = self.scale * (F::ONE - r);
        let at_one = self.scale * r;
        let inverse = at_one.inverse();
        let [v0, v1, v2] = match inverse.is_some() {
            true => self.node_sums::<false>(k, &eq_lo, &eq_hi),
            false => self.node_sums::<true>(k, &eq_lo, &eq_hi),
        };
        let v1 = match inverse {
            Some(inverse) => (self.claim - at_zero * v0) * inverse,
            None => v1,
        };
        let v3 = v0 + (v2 - v1).double() + (v2 - v1);
        message[0] += self.weight * at_zero * v0;
        message[1] += self.weight * (at_one.double() - at_zero) * v2;
        message[2] += self.weight * (at_one.double() + at_one - at_zero.double()) * v3;
        self.sums = [v0, v1, v2];
    }

    /// `v(0)`, `v(2)`, and `v(1)` when `ONE`, of own round `k`: from the borrowed tables in the
    /// first own round, from the bound ones after.
    fn node_sums<const ONE: bool>(&mut self, k: usize, eq_lo: &[F], eq_hi: &[F]) -> [F; 3] {
        if let Some(window) = &self.window
            && k < window.depth
        {
            return window.sums(&self.point[k + 1..]);
        }
        let base = |w: F, b: F::Base| w.mul_by_base(b);
        let ext = |w: F, x: F| w * x;
        if k > 0 {
            let b = &self.bound;
            return match self.tables {
                LevelTables::Product { .. } => {
                    product_sums::<F, F, ONE>(&b[0], &b[1], eq_lo, eq_hi, ext)
                }
                LevelTables::Fraction {
                    numerators, lambda, ..
                } => {
                    let n = match numerators {
                        LevelNumerators::Unit => None,
                        _ => Some((&b[0][..], &b[1][..])),
                    };
                    fraction_sums::<F, F, ONE>(n, &b[2], &b[3], *lambda, eq_lo, eq_hi, ext)
                }
            };
        }
        match self.tables {
            LevelTables::Product { even, odd } => {
                product_sums::<F, F::Base, ONE>(even, odd, eq_lo, eq_hi, base)
            }
            LevelTables::Fraction {
                numerators,
                d_even,
                d_odd,
                lambda,
                ..
            } => match numerators {
                LevelNumerators::Unit => {
                    fraction_sums::<F, F, ONE>(None, d_even, d_odd, *lambda, eq_lo, eq_hi, ext)
                }
                LevelNumerators::Base { even, odd } => fraction_sums::<F, F::Base, ONE>(
                    Some((even, odd)),
                    d_even,
                    d_odd,
                    *lambda,
                    eq_lo,
                    eq_hi,
                    base,
                ),
                LevelNumerators::Ext { even, odd } => fraction_sums::<F, F, ONE>(
                    Some((even, odd)),
                    d_even,
                    d_odd,
                    *lambda,
                    eq_lo,
                    eq_hi,
                    ext,
                ),
            },
        }
    }

    /// After round `j`'s challenge `c`: the claim `p(c)` and `s·ℓ(c)` of an own round, and its
    /// tables bound at `c`.
    fn bind(&mut self, j: usize, c: F, half: F) {
        if j < self.pad {
            return;
        }
        let k = j - self.pad;
        let r = self.point[k];
        let at_zero = self.scale * (F::ONE - r);
        let at_one = self.scale * r;
        let [v0, v1, v2] = self.sums;
        // Newton's form through 0, 1, 2: v(c) = v0 + c·(v1 − v0) + c(c − 1)/2·(v2 − 2v1 + v0).
        let second = v2 - v1.double() + v0;
        let v_c = v0 + c * (v1 - v0) + c * (c - F::ONE) * half * second;
        let line_c = at_zero + c * (at_one - at_zero);
        self.claim = line_c * v_c;
        self.scale = line_c;
        if let Some(window) = &mut self.window
            && k < window.depth
        {
            window.bind(c, half);
            if k + 1 == window.depth
                && let LevelTables::Fraction { raw: Some(raw), .. } = self.tables
            {
                self.bound = raw.bind(&window.challenges);
            }
            return;
        }
        if k == 0 {
            self.bind_first(c);
        } else {
            for table in &mut self.bound {
                if !table.is_empty() {
                    bind_in_place(table, c);
                }
            }
        }
    }

    /// Binds the borrowed tables at `c` into the instance's own.
    fn bind_first(&mut self, c: F) {
        let half = 1usize << (self.vars() - 1);
        let reserved = || SecretBuf::with_capacity(half);
        let mut bound: Vec<SecretBuf<F>> = Vec::with_capacity(4);
        match self.tables {
            LevelTables::Product { even, odd } => {
                for table in [even, odd] {
                    let mut out = reserved();
                    bind_base(table, c, &mut out);
                    bound.push(out);
                }
            }
            LevelTables::Fraction {
                numerators,
                d_even,
                d_odd,
                ..
            } => {
                match numerators {
                    LevelNumerators::Unit => bound.extend([SecretBuf::new(), SecretBuf::new()]),
                    LevelNumerators::Base { even, odd } => {
                        for table in [even, odd] {
                            let mut out = reserved();
                            bind_base(table, c, &mut out);
                            bound.push(out);
                        }
                    }
                    LevelNumerators::Ext { even, odd } => {
                        for table in [even, odd] {
                            let mut out = reserved();
                            bind_ext(table, c, &mut out);
                            bound.push(out);
                        }
                    }
                }
                for table in [d_even, d_odd] {
                    let mut out = reserved();
                    bind_ext(table, c, &mut out);
                    bound.push(out);
                }
            }
        }
        self.bound = bound;
    }

    /// Appends the instance's final evaluations, its tables' values at its point: the bound
    /// tables' single entries, the unbound tables' of an instance on no variable, and 1 for
    /// numerators of 1.
    fn final_evals(&self, out: &mut Vec<F>) {
        if self.vars() == 0
            && let LevelTables::Fraction { raw: Some(raw), .. } = self.tables
        {
            let (n0, d0) = raw.value(0);
            let (n1, d1) = raw.value(1);
            out.extend([n0, n1, d0, d1]);
            return;
        }

        let single = |bound: Option<&SecretBuf<F>>, unbound: F| match bound {
            Some(table) => table.first().copied().unwrap_or(F::ZERO),
            None => unbound,
        };
        let bound = |i: usize| (self.vars() > 0).then(|| &self.bound[i]);
        let first_base = |t: &[F::Base]| lift_base::<F>(t).next().unwrap_or(F::ZERO);
        let first_ext = |t: &[F]| t.first().copied().unwrap_or(F::ZERO);
        match self.tables {
            LevelTables::Product { even, odd } => {
                out.push(single(bound(0), first_base(even)));
                out.push(single(bound(1), first_base(odd)));
            }
            LevelTables::Fraction {
                numerators,
                d_even,
                d_odd,
                ..
            } => {
                match numerators {
                    LevelNumerators::Unit => out.extend([F::ONE, F::ONE]),
                    LevelNumerators::Base { even, odd } => {
                        out.push(single(bound(0), first_base(even)));
                        out.push(single(bound(1), first_base(odd)));
                    }
                    LevelNumerators::Ext { even, odd } => {
                        out.push(single(bound(0), first_ext(even)));
                        out.push(single(bound(1), first_ext(odd)));
                    }
                }
                out.push(single(bound(2), first_ext(d_even)));
                out.push(single(bound(3), first_ext(d_odd)));
            }
        }
    }
}

/// Proves one tree level over `instances`, with `γ` already squeezed, under the labels of
/// `prefix`: the same rounds and final evaluations, absorbed and squeezed in the same order, as
/// the batch prover of degree 3 over `[eq(r_i, ·), …]` instances with `eq` a public table.
///
/// Each instance's tables hold `2^{n_i}` entries, `n_i` the length of its point, and its claim is
/// its sum over its own hypercube; the caller checks the shapes.
pub(crate) fn prove_level<F, T>(
    transcript: &mut T,
    prefix: Prefix,
    instances: &[LevelInstance<'_, F>],
    gamma: F,
) -> ProvenBatch<F>
where
    F: crate::BenchField,
    T: SumcheckTranscript<F> + ?Sized,
{
    let mut labels = Labels::new(prefix);
    let vars: Vec<usize> = instances.iter().map(|i| i.point.len()).collect();
    let n_max = vars.iter().copied().max().unwrap_or(0);
    let half = F::ONE.double().inverse().unwrap_or(F::ZERO);
    let mut weight = F::ONE;
    let mut states = Vec::with_capacity(instances.len());
    for (instance, &n) in instances.iter().zip(&vars) {
        states.push(Split {
            tables: &instance.tables,
            point: instance.point,
            pad: n_max - n,
            weight,
            claim: instance.claim,
            scale: F::ONE,
            sums: [F::ZERO; 3],
            bound: Vec::new(),
            window: match &instance.tables {
                LevelTables::Fraction {
                    raw: Some(raw),
                    lambda,
                    ..
                } if !instance.point.is_empty() => Some(crate::coeff::Window::new(
                    raw,
                    instance.point,
                    *lambda,
                    crate::width().min(instance.point.len()),
                )),
                _ => None,
            },
        });
        weight *= gamma;
    }

    let mut rounds = Vec::with_capacity(n_max);
    let mut point = Vec::with_capacity(n_max);
    for j in 0..n_max {
        let mut message = [F::ZERO; 3];
        for state in &mut states {
            state.add_round(j, &mut message);
        }
        transcript.absorb_message(labels.round(j), &message);
        let c = transcript.challenge(labels.challenge(j));
        for state in &mut states {
            state.bind(j, c, half);
        }
        rounds.push(message.to_vec());
        point.push(c);
    }

    let mut final_evals = Vec::new();
    let mut evals = Vec::with_capacity(states.len());
    for state in &states {
        let start = final_evals.len();
        state.final_evals(&mut final_evals);
        evals.push(start..final_evals.len());
    }
    transcript.absorb_message(labels.evals(), &final_evals);
    ProvenBatch::from_parts(
        SumcheckProof {
            rounds,
            final_evals,
        },
        gamma,
        point,
        vars,
        evals,
    )
}
