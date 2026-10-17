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
pub(crate) enum LevelNumerators<'a, F: PolyField> {
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
pub(crate) enum LevelTables<'a, F: PolyField> {
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
    },
}

/// One instance of a tree level: its point `r_i`, lowest variable first; its claim `c_i`, the
/// sum it proves; and the level below.
pub(crate) struct LevelInstance<'a, F: PolyField> {
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
    F: Field,
    N: Copy + Add<Output = N> + Sub<Output = N>,
{
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
fn lift_base<F: PolyField>(values: &[F::Base]) -> impl Iterator<Item = F> + '_ {
    values.iter().map(|&b| F::ZERO.add_base(b))
}

/// `Σ_x eq(point, x)·even[x]·odd[x]`: a product forest's root claim at its root point, the sum
/// its first level proves, from the halves of its level 1, with `eq` split as the rounds split it.
pub(crate) fn product_claim<F: PolyField>(point: &[F], even: &[F::Base], odd: &[F::Base]) -> F {
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
struct Split<'a, F: PolyField> {
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
}

/// The bound tables clear themselves; the node sums and the claim, which no message carries alone,
/// are cleared here (05 §3 OBS-6).
impl<F: PolyField> Drop for Split<'_, F> {
    fn drop(&mut self) {
        self.sums.zeroize();
        self.claim.zeroize();
    }
}

impl<'a, F: PolyField> Split<'a, F> {
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
    fn node_sums<const ONE: bool>(&self, k: usize, eq_lo: &[F], eq_hi: &[F]) -> [F; 3] {
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
    F: PolyField,
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

#[cfg(test)]
mod tests {
    //! The split-`eq` kernel against the batch prover over `[eq, …]` instances, the reference it
    //! replaces: the same proof, point and transcript state for products on 0 to 12 variables,
    //! fraction trees with every kind of numerator, front-padded batches of both, and the three
    //! cases where honest challenges land with probability about `2^-128`: a point coordinate of
    //! 0 (the fallback sums `v(1)`), a coordinate of 1 (no fallback: `s·ℓ(0) = 0`), and a
    //! challenge that zeroes `s` (the fallback in every later round). A scripted transcript
    //! drives the last.

    use tessera_codec::Ascii;
    use tessera_field::{Field, Goldilocks, GoldilocksExt2, PrimeField, QuadraticExtension};
    use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
    use tessera_poly::PolyField;

    use super::{LevelInstance, LevelNumerators, LevelTables, product_claim, prove_level};
    use crate::contract::eq_weights;
    use crate::prover::prove_batch_after_gamma;
    use crate::{Instance, Prefix, ProvenBatch, SumcheckTranscript, Table, Term};

    type Fp = Goldilocks;
    type Fp2 = GoldilocksExt2;

    struct Rng(u64);

    impl Rng {
        fn u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 1
        }

        fn fp(&mut self) -> Fp {
            Fp::from_u64_reduced(self.u64())
        }

        fn fp2(&mut self) -> Fp2 {
            Fp2::new(self.fp(), self.fp())
        }

        fn fps(&mut self, n: usize) -> Vec<Fp> {
            (0..n).map(|_| self.fp()).collect()
        }

        fn fp2s(&mut self, n: usize) -> Vec<Fp2> {
            (0..n).map(|_| self.fp2()).collect()
        }
    }

    /// RFC-0004's transcript, with some challenges replaced by a script, its state moving as if
    /// they were not.
    struct Scripted {
        inner: Transcript,
        script: Vec<(String, Fp2)>,
    }

    impl Scripted {
        fn new(script: Vec<(String, Fp2)>) -> Self {
            let context = SessionContext {
                protocol: Ascii::new("tessera-protocol/1").expect("ASCII"),
                mode: Mode::NonInteractive,
                params: ParamSetId(Ascii::new("test-small/1").expect("ASCII")),
                statement_digest: [0x25; 32],
                session_id: [8; 16],
            };
            Self {
                inner: Transcript::new(&context),
                script,
            }
        }
    }

    impl SumcheckTranscript<Fp2> for Scripted {
        fn absorb_message(&mut self, label: &str, message: &[Fp2]) {
            self.inner.absorb_message(label, message);
        }

        fn challenge(&mut self, label: &str) -> Fp2 {
            let drawn = self.inner.challenge(label);
            self.script
                .iter()
                .find(|(scripted, _)| scripted == label)
                .map_or(drawn, |&(_, value)| value)
        }
    }

    /// The tables of one instance, owned.
    enum Owned {
        Product(Vec<Fp>, Vec<Fp>),
        Unit(Vec<Fp2>, Vec<Fp2>),
        Base(Vec<Fp>, Vec<Fp>, Vec<Fp2>, Vec<Fp2>),
        Ext(Vec<Fp2>, Vec<Fp2>, Vec<Fp2>, Vec<Fp2>),
    }

    struct Case {
        point: Vec<Fp2>,
        tables: Owned,
    }

    impl Case {
        fn product(rng: &mut Rng, point: Vec<Fp2>) -> Self {
            let len = 1 << point.len();
            let tables = Owned::Product(rng.fps(len), rng.fps(len));
            Self { point, tables }
        }

        fn fraction(rng: &mut Rng, point: Vec<Fp2>, kind: usize) -> Self {
            let len = 1 << point.len();
            let tables = match kind {
                0 => Owned::Unit(rng.fp2s(len), rng.fp2s(len)),
                1 => Owned::Base(rng.fps(len), rng.fps(len), rng.fp2s(len), rng.fp2s(len)),
                _ => Owned::Ext(rng.fp2s(len), rng.fp2s(len), rng.fp2s(len), rng.fp2s(len)),
            };
            Self { point, tables }
        }

        /// The sum the instance proves, by brute force over the hypercube.
        fn claim(&self, lambda: Fp2) -> Fp2 {
            let eq = eq_weights(&self.point);
            let lift = |b: Fp| Fp2::ZERO.add_base(b);
            let at = |x: usize| -> Fp2 {
                match &self.tables {
                    Owned::Product(e, o) => lift(e[x] * o[x]),
                    Owned::Unit(d0, d1) => d1[x] + d0[x] + lambda * d0[x] * d1[x],
                    Owned::Base(n0, n1, d0, d1) => {
                        lift(n0[x]) * d1[x] + lift(n1[x]) * d0[x] + lambda * d0[x] * d1[x]
                    }
                    Owned::Ext(n0, n1, d0, d1) => {
                        n0[x] * d1[x] + n1[x] * d0[x] + lambda * d0[x] * d1[x]
                    }
                }
            };
            (0..eq.len())
                .map(|x| eq[x] * at(x))
                .fold(Fp2::ZERO, |a, b| a + b)
        }

        fn level(&self, lambda: Fp2) -> LevelInstance<'_, Fp2> {
            let tables = match &self.tables {
                Owned::Product(e, o) => LevelTables::Product { even: e, odd: o },
                Owned::Unit(d0, d1) => LevelTables::Fraction {
                    numerators: LevelNumerators::Unit,
                    d_even: d0,
                    d_odd: d1,
                    lambda,
                },
                Owned::Base(n0, n1, d0, d1) => LevelTables::Fraction {
                    numerators: LevelNumerators::Base { even: n0, odd: n1 },
                    d_even: d0,
                    d_odd: d1,
                    lambda,
                },
                Owned::Ext(n0, n1, d0, d1) => LevelTables::Fraction {
                    numerators: LevelNumerators::Ext { even: n0, odd: n1 },
                    d_even: d0,
                    d_odd: d1,
                    lambda,
                },
            };
            LevelInstance {
                point: &self.point,
                claim: self.claim(lambda),
                tables,
            }
        }
    }

    /// `eq·(N_0·D_1 + N_1·D_0 + λ·D_0·D_1)` over `[eq, N_0, N_1, D_0, D_1]`, reading no numerator
    /// when they are 1.
    fn fraction_terms(lambda: Fp2, unit: bool) -> Vec<Term<Fp2>> {
        let (n0_d1, n1_d0) = match unit {
            true => (vec![0, 4], vec![0, 3]),
            false => (vec![0, 1, 4], vec![0, 2, 3]),
        };
        vec![
            Term::new(Fp2::ONE, n0_d1),
            Term::new(Fp2::ONE, n1_d0),
            Term::new(lambda, vec![0, 3, 4]),
        ]
    }

    /// Both provers on the same cases and script; the proofs, points, evaluations and transcript
    /// states must agree.
    fn same_as_the_batch_prover(cases: &[Case], script: Vec<(String, Fp2)>) {
        let (mut ours, mut theirs) = (Scripted::new(script.clone()), Scripted::new(script));
        let gamma = ours.challenge("tree/0/gamma");
        assert_eq!(theirs.challenge("tree/0/gamma"), gamma);
        let lambda = ours.challenge("tree/0/lambda");
        assert_eq!(theirs.challenge("tree/0/lambda"), lambda);

        let levels: Vec<LevelInstance<'_, Fp2>> = cases.iter().map(|c| c.level(lambda)).collect();
        let proven = prove_level(&mut ours, Prefix::Tree(0), &levels, gamma);

        let eqs: Vec<Vec<Fp2>> = cases.iter().map(|c| eq_weights(&c.point)).collect();
        let ones: Vec<Vec<Fp>> = cases
            .iter()
            .map(|c| vec![Fp::ONE; 1 << c.point.len()])
            .collect();
        let instances: Vec<Instance<'_, Fp2>> = cases
            .iter()
            .zip(eqs.iter().zip(&ones))
            .map(|(case, (eq, ones))| {
                let eq = Table::ext(eq).public();
                match &case.tables {
                    Owned::Product(e, o) => Instance::new(vec![eq, Table::base(e), Table::base(o)]),
                    Owned::Unit(d0, d1) => Instance::sum_of_products(
                        vec![
                            eq,
                            Table::base(ones),
                            Table::base(ones),
                            Table::ext(d0),
                            Table::ext(d1),
                        ],
                        fraction_terms(lambda, true),
                    ),
                    Owned::Base(n0, n1, d0, d1) => Instance::sum_of_products(
                        vec![
                            eq,
                            Table::base(n0),
                            Table::base(n1),
                            Table::ext(d0),
                            Table::ext(d1),
                        ],
                        fraction_terms(lambda, false),
                    ),
                    Owned::Ext(n0, n1, d0, d1) => Instance::sum_of_products(
                        vec![
                            eq,
                            Table::ext(n0),
                            Table::ext(n1),
                            Table::ext(d0),
                            Table::ext(d1),
                        ],
                        fraction_terms(lambda, false),
                    ),
                }
            })
            .collect();
        let reference: ProvenBatch<Fp2> =
            prove_batch_after_gamma(&mut theirs, Prefix::Tree(0), 3, &instances, gamma)
                .expect("valid instances");

        assert_eq!(proven.proof(), reference.proof());
        assert_eq!(proven.point(), reference.point());
        for i in 0..cases.len() {
            assert_eq!(proven.instance_evals(i), reference.instance_evals(i), "{i}");
            assert_eq!(proven.instance_point(i), reference.instance_point(i), "{i}");
        }
        assert_eq!(ours.challenge("after"), theirs.challenge("after"));
    }

    #[test]
    fn products_on_0_to_12_variables_match_the_batch_prover() {
        let mut rng = Rng(0x2580_0001);
        for n in 0..=12 {
            let point = rng.fp2s(n);
            same_as_the_batch_prover(&[Case::product(&mut rng, point)], Vec::new());
        }
    }

    #[test]
    fn fraction_trees_of_every_numerator_kind_match_the_batch_prover() {
        let mut rng = Rng(0x2580_0002);
        for kind in 0..3 {
            for n in 0..=8 {
                let point = rng.fp2s(n);
                same_as_the_batch_prover(&[Case::fraction(&mut rng, point, kind)], Vec::new());
            }
        }
    }

    #[test]
    fn front_padded_batches_of_both_kinds_match_the_batch_prover() {
        let mut rng = Rng(0x2580_0003);
        let shapes = [
            (3, None),
            (7, Some(2)),
            (0, None),
            (5, Some(0)),
            (1, Some(1)),
            (6, None),
        ];
        let cases: Vec<Case> = shapes
            .iter()
            .map(|&(n, fraction)| {
                let point = rng.fp2s(n);
                match fraction {
                    Some(kind) => Case::fraction(&mut rng, point, kind),
                    None => Case::product(&mut rng, point),
                }
            })
            .collect();
        same_as_the_batch_prover(&cases, Vec::new());
    }

    /// `ℓ(1) = s·r_k = 0` at the coordinate of 0: the kernel sums `v(1)` from the tables.
    #[test]
    fn a_zero_coordinate_takes_the_fallback() {
        let mut rng = Rng(0x2580_0004);
        for kind in [None, Some(0), Some(1), Some(2)] {
            let mut point = rng.fp2s(6);
            point[2] = Fp2::ZERO;
            let case = match kind {
                Some(kind) => Case::fraction(&mut rng, point, kind),
                None => Case::product(&mut rng, point),
            };
            same_as_the_batch_prover(&[case], Vec::new());
        }
    }

    /// `ℓ(0) = s·(1 − r_k) = 0` at the coordinate of 1: the recovery divides by `s`, no fallback.
    #[test]
    fn a_coordinate_of_one_divides_by_s() {
        let mut rng = Rng(0x2580_0005);
        let mut point = rng.fp2s(6);
        point[3] = Fp2::ONE;
        same_as_the_batch_prover(&[Case::product(&mut rng, point)], Vec::new());
    }

    /// A challenge `c_k = (r_k − 1)/(2r_k − 1)` makes `ℓ_k(c_k) = 0`, so `s = 0` from round
    /// `k + 1` on: the fallback runs in every later round, in a batch whose other instance does
    /// not take it.
    #[test]
    fn a_challenge_that_zeroes_s_takes_the_fallback_after() {
        let mut rng = Rng(0x2580_0006);
        let point = rng.fp2s(7);
        let k = 2;
        let r = point[k];
        let zeroing = (r - Fp2::ONE) * (r.double() - Fp2::ONE).inverse().expect("non-zero");
        let script = vec![(format!("tree/0/r/{k}"), zeroing)];
        let other = rng.fp2s(5);
        let cases = [
            Case::product(&mut rng, point),
            Case::fraction(&mut rng, other, 1),
        ];
        same_as_the_batch_prover(&cases, script);
    }

    /// The root claim of a forest, from its level 1's halves, is the sum its first level proves.
    #[test]
    fn a_forests_root_claim_is_its_first_levels_sum() {
        let mut rng = Rng(0x2580_0007);
        for s in 0..=9 {
            let point = rng.fp2s(s);
            let (even, odd) = (rng.fps(1 << s), rng.fps(1 << s));
            let eq = eq_weights(&point);
            let expected = (0..1 << s)
                .map(|y| eq[y].mul_by_base(even[y] * odd[y]))
                .fold(Fp2::ZERO, |a, b| a + b);
            assert_eq!(product_claim(&point, &even, &odd), expected, "{s}");
        }
    }
}
