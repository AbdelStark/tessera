// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0054's merge: the sumcheck that reduces a grouped leaf stage's claims on each merged
//! polynomial, at the points of the groups that read it, to one claim at one point.
//!
//! - **The polynomials.** [`MergePolynomial`] `P_i` has `n_i` variables, among them its group
//!   variables `S_i`: the coordinates a group fixes to its own bits.
//! - **The claims.** Group `h`'s claim `k` is `P̃_i(z ‖ s) = v` ([`MergeClaim`]): `z` over the
//!   variables outside `S_i`, `s` its bits on `S_i`.
//! - **The relation.** The merge proves `Σ_h β^h·Σ_k γ_h^k·2^{n_max − n_i}·v_{k,h}` of
//!   `Σ_x Σ_i W_i(x)·P̃_i(x)`, front-padded, with
//!   `W_i(x) = Σ_{(h, k) on P_i} β^h·γ_h^k·eq(z_{k,h}, x_{S̄_i})·eq(s_{k,h}, x_{S_i})`.
//! - **The order.** Each polynomial's variables outside `S_i` are bound first, lowest first, then
//!   those in `S_i`, lowest first. A polynomial of `n_i` variables takes the last `n_i`
//!   challenges.
//! - **The messages.** Degree 2, RFC-0010's rounds under `merge/round/{j}` and `merge/r/{j}`, then
//!   every `P̃_i(r_i)` under `merge/evals`, in polynomial order. The weights are public: the
//!   verifier computes every `W_i(r_i)` itself.
//!
//! Two provers send the same messages:
//! - **[`prove_merge`].** It holds every polynomial in memory and runs the batch engine, with
//!   instance `i` the product `W_i·P̃_i` at weight 1.
//! - **[`prove_merge_two_pass`].** It answers the first `j*` rounds from tables kept in each
//!   group's pass ([`MergeTables`]), then regenerates every polynomial once, folds the variables
//!   those rounds bound, and runs the remaining rounds in memory (RFC-0054 'The prover in two
//!   passes').
//!
//! The polynomials and the groups' blocks are secret-class tensors (05 §3): they come in their
//! [`Secret`], and what the provers derive from them lives in [`SecretBuf`]s. The weights derive
//! from public points, challenges and bits.

use core::fmt;

use tessera_codec::{Secret, SecretBuf};
use tessera_poly::{PolyField, eq, eq_table};

use crate::prover::{Instance, Labels, Prefix, ProvenBatch, Table, prove_batch_unweighted};
use crate::{SumcheckError, SumcheckProof, SumcheckTranscript};

/// The most variables a merged polynomial may have: its table and its weights must fit a `usize`
/// index, and `2^{n_max − n_i}` the prime subfield.
pub const MAX_MERGE_VARS: usize = 40;

/// A merged polynomial's shape: `vars` variables, `group_vars` of them fixed by a group, as
/// positions from 0 for the lowest, strictly increasing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MergePolynomial {
    /// `n_i`.
    pub vars: usize,
    /// `S_i`, the positions of the group variables.
    pub group_vars: Vec<usize>,
}

impl MergePolynomial {
    /// The variables outside `S_i`, `m_i = n_i − |S_i|`.
    pub fn free_vars(&self) -> usize {
        self.vars - self.group_vars.len()
    }

    /// The binding order: the positions outside `S_i`, ascending, then those in `S_i`.
    fn order(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.vars)
            .filter(|v| self.group_vars.binary_search(v).is_err())
            .collect();
        order.extend(&self.group_vars);
        order
    }
}

/// One group's claim on a polynomial: `P̃_i(z ‖ s) = value`.
#[derive(Clone, PartialEq, Eq)]
pub struct MergeClaim<F> {
    /// The polynomial, an index into the merge's polynomials.
    pub polynomial: usize,
    /// `z`: one coordinate per variable outside `S_i`, in increasing position.
    pub point: Vec<F>,
    /// `s`: bit `u` is the value of the `u`-th group variable, in increasing position.
    pub group_bits: u64,
    /// The claimed value.
    pub value: F,
}

/// The polynomial, the point's length and the bits: points and values are protocol-class.
impl<F> fmt::Debug for MergeClaim<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MergeClaim")
            .field("polynomial", &self.polynomial)
            .field("point", &self.point.len())
            .field("group_bits", &self.group_bits)
            .finish_non_exhaustive()
    }
}

/// A merge's public inputs, checked: its polynomials, each group's claims in order, `γ_h` per
/// group and `β`.
#[derive(Clone)]
pub struct Merge<F> {
    polynomials: Vec<MergePolynomial>,
    claims: Vec<Vec<MergeClaim<F>>>,
    gammas: Vec<F>,
    beta: F,
    n_max: usize,
}

/// Shapes and counts: claims and challenges are protocol-class.
impl<F> fmt::Debug for Merge<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let claims: Vec<usize> = self.claims.iter().map(Vec::len).collect();
        f.debug_struct("Merge")
            .field("polynomials", &self.polynomials)
            .field("claims", &claims)
            .finish_non_exhaustive()
    }
}

impl<F: PolyField> Merge<F> {
    /// The merge of `claims[h]`, group `h`'s claims in its order `k`, on `polynomials`, with
    /// `gammas[h]` as `γ_h` and `beta` as `β`.
    ///
    /// # Errors
    ///
    /// [`SumcheckError::Merge`] for no polynomial, a polynomial of more than
    /// [`MAX_MERGE_VARS`] variables or with group variables that are not strictly increasing
    /// positions among them, a `γ_h` count other than the groups', or a claim on no polynomial,
    /// with a point of another size than the polynomial's free variables, or with bits beyond its
    /// group variables.
    pub fn new(
        polynomials: Vec<MergePolynomial>,
        claims: Vec<Vec<MergeClaim<F>>>,
        gammas: Vec<F>,
        beta: F,
    ) -> Result<Self, SumcheckError> {
        let bad = |rule| Err(SumcheckError::Merge { rule });
        if polynomials.is_empty() {
            return bad("a merge has a polynomial");
        }
        for p in &polynomials {
            if p.vars > MAX_MERGE_VARS {
                return bad("a polynomial has at most MAX_MERGE_VARS variables");
            }
            let increasing = p.group_vars.windows(2).all(|w| w[0] < w[1]);
            if !increasing || p.group_vars.last().is_some_and(|&v| v >= p.vars) {
                return bad("group variables are increasing positions among the variables");
            }
        }
        if gammas.len() != claims.len() {
            return bad("one gamma per group");
        }
        for claim in claims.iter().flatten() {
            let Some(p) = polynomials.get(claim.polynomial) else {
                return bad("a claim is on a polynomial of the merge");
            };
            if claim.point.len() != p.free_vars() {
                return bad("a claim's point has one coordinate per free variable");
            }
            if claim.group_bits >> p.group_vars.len() != 0 {
                return bad("a claim's bits fit its polynomial's group variables");
            }
        }
        let n_max = polynomials.iter().map(|p| p.vars).max().unwrap_or(0);
        Ok(Self {
            polynomials,
            claims,
            gammas,
            beta,
            n_max,
        })
    }

    /// The polynomials.
    pub fn polynomials(&self) -> &[MergePolynomial] {
        &self.polynomials
    }

    /// `n_max`, the merge's rounds.
    pub fn rounds(&self) -> usize {
        self.n_max
    }

    /// Each group's claims, in its order `k`.
    pub fn claims(&self) -> &[Vec<MergeClaim<F>>] {
        &self.claims
    }

    /// `γ_h` per group.
    pub fn gammas(&self) -> &[F] {
        &self.gammas
    }

    /// `β`.
    pub fn beta(&self) -> F {
        self.beta
    }

    /// Every claim with its weight `β^h·γ_h^k`.
    fn weighted(&self) -> impl Iterator<Item = (F, &MergeClaim<F>)> {
        let mut outer = F::ONE;
        self.claims
            .iter()
            .zip(&self.gammas)
            .flat_map(move |(group, &gamma)| {
                let beta_h = outer;
                outer *= self.beta;
                let mut inner = F::ONE;
                group.iter().map(move |claim| {
                    let weight = beta_h * inner;
                    inner *= gamma;
                    (weight, claim)
                })
            })
    }

    /// The claimed sum `Σ_h β^h·Σ_k γ_h^k·2^{n_max − n_i}·v_{k,h}`, the first round's claim.
    pub fn claimed_sum(&self) -> F {
        self.weighted()
            .map(|(w, c)| w * c.value * pow2::<F>(self.n_max - self.polynomials[c.polynomial].vars))
            .fold(F::ZERO, |a, b| a + b)
    }

    /// `c_i = Σ_{(h, k) on P_i} β^h·γ_h^k·v_{k,h}`: polynomial `i`'s sum over its own hypercube,
    /// when its claims are true.
    fn claimed_sums(&self) -> Vec<F> {
        let mut sums = vec![F::ZERO; self.polynomials.len()];
        for (w, c) in self.weighted() {
            sums[c.polynomial] += w * c.value;
        }
        sums
    }

    /// `W_i` in binding order: `2^{n_i}` entries, the free variables lowest.
    ///
    /// # Errors
    ///
    /// [`SumcheckError::Merge`] for a polynomial the merge does not have, or a table that
    /// cannot be allocated.
    pub fn weights(&self, i: usize) -> Result<Vec<F>, SumcheckError> {
        let bad = |rule| SumcheckError::Merge { rule };
        let p = self
            .polynomials
            .get(i)
            .ok_or(bad("the polynomial is one of the merge's"))?;
        let m = p.free_vars();
        let len = 1usize << p.vars;
        let mut table = Vec::new();
        table
            .try_reserve_exact(len)
            .map_err(|_| bad("a polynomial's weights fit in memory"))?;
        table.resize(len, F::ZERO);
        for (w, c) in self.weighted().filter(|(_, c)| c.polynomial == i) {
            let eq = eq_table(&c.point).map_err(|_| SumcheckError::Merge {
                rule: "a claim's eq table fits in memory",
            })?;
            let base = usize::try_from(c.group_bits).unwrap_or(usize::MAX) << m;
            for (entry, e) in table[base..base + (1 << m)].iter_mut().zip(eq) {
                *entry += w * e;
            }
        }
        Ok(table)
    }

    /// `W_i(r_i)` at its point in binding order, from the claims alone: what a verifier
    /// computes. `None` for a polynomial the merge does not have or a point of another length.
    pub fn weight_at(&self, i: usize, point: &[F]) -> Option<F> {
        let p = self.polynomials.get(i)?;
        if point.len() != p.vars {
            return None;
        }
        let m = p.free_vars();
        let mut total = F::ZERO;
        for (w, c) in self.weighted().filter(|(_, c)| c.polynomial == i) {
            let bits: Vec<F> = (0..p.group_vars.len())
                .map(|u| match (c.group_bits >> u) & 1 {
                    1 => F::ONE,
                    _ => F::ZERO,
                })
                .collect();
            let free = eq(&c.point, &point[..m]).ok()?;
            let fixed = eq(&bits, &point[m..]).ok()?;
            total += w * free * fixed;
        }
        Some(total)
    }
}

/// `2^e` in the prime subfield, `e ≤ MAX_MERGE_VARS`.
fn pow2<F: PolyField>(e: usize) -> F {
    (0..e).fold(F::ONE, |x, _| x.double())
}

/// `table` in binding order: entry `y` is the entry whose variable `order[t]` is bit `t` of `y`.
/// The table is checked non-empty, `2^{n_i}` entries.
fn permuted<B: Copy>(table: &[B], order: &[usize]) -> SecretBuf<B> {
    let Some(&first) = table.first() else {
        return SecretBuf::new();
    };
    let mut out = SecretBuf::filled(first, table.len());
    for (y, slot) in out.iter_mut().enumerate() {
        let x = order
            .iter()
            .enumerate()
            .fold(0usize, |x, (t, &v)| x | (((y >> t) & 1) << v));
        *slot = table[x];
    }
    out
}

/// Checks one table per polynomial, each of `2^{n_i}` entries.
#[allow(clippy::disallowed_methods)]
fn checked_tables<'t, F: PolyField>(
    merge: &Merge<F>,
    tables: &[&'t Secret<Vec<F::Base>>],
) -> Result<Vec<&'t [F::Base]>, SumcheckError> {
    if tables.len() != merge.polynomials.len() {
        return Err(SumcheckError::Merge {
            rule: "one table per polynomial",
        });
    }
    merge
        .polynomials
        .iter()
        .zip(tables)
        .map(|(p, t)| {
            let t = t.expose_secret().as_slice();
            match t.len() == 1 << p.vars {
                true => Ok(t),
                false => Err(SumcheckError::Merge {
                    rule: "a polynomial's table has 2^n_i entries",
                }),
            }
        })
        .collect()
}

/// RFC-0054's merge with every polynomial in memory: the batch engine over the instances
/// `W_i·P̃_i`, each at weight 1, under `merge/…`. `merge/beta` and the groups' `γ_h` come before
/// it, from the caller.
///
/// # Errors
///
/// [`SumcheckError::Merge`] for a table count other than the polynomials' or a table of another
/// length than `2^{n_i}`, before the transcript moves.
pub fn prove_merge<F, T>(
    transcript: &mut T,
    merge: &Merge<F>,
    tables: &[&Secret<Vec<F::Base>>],
) -> Result<ProvenBatch<F>, SumcheckError>
where
    F: PolyField,
    T: SumcheckTranscript<F> + ?Sized,
{
    let tables = checked_tables(merge, tables)?;
    let mut weights = Vec::with_capacity(tables.len());
    let mut ordered = Vec::with_capacity(tables.len());
    for (i, (p, table)) in merge.polynomials.iter().zip(&tables).enumerate() {
        weights.push(merge.weights(i)?);
        ordered.push(permuted(table, &p.order()));
    }
    let instances: Vec<Instance<'_, F>> = weights
        .iter()
        .zip(&ordered)
        .map(|(w, t)| Instance::new(vec![Table::ext(w).public(), Table::base(t)]))
        .collect();
    prove_batch_unweighted(transcript, Prefix::Merge, 2, &instances)
}

/// The tables the groups' pass keeps for the first `j*` rounds of [`prove_merge_two_pass`].
///
/// For a class `c` of group `h`, claims on polynomials of one shape at one point, write
/// `Q_{c,h} = Σ_{k∈c} γ_h^k·P_{i(k)}(·, s)` over the free variables. Own round `t` of the class
/// reads `T_t(a, b) = Q̃_{c,h}(a, b, z_{>t})`, `2^{t+1}` entries, kept for each `t` below the
/// class's table rounds.
pub struct MergeTables<F: PolyField> {
    j_star: usize,
    classes: Vec<Class<F>>,
    kept: Vec<bool>,
}

/// One class's tables.
struct Class<F: PolyField> {
    group: usize,
    point: Vec<F>,
    padding: usize,
    tables: Vec<SecretBuf<F>>,
}

impl<F: PolyField> MergeTables<F> {
    /// Tables for the first `j_star` rounds.
    ///
    /// # Errors
    ///
    /// [`SumcheckError::Merge`] when some polynomial's free variables end before round
    /// `j_star`: past them a round binds group variables, whose other groups' blocks no group
    /// holds.
    pub fn new(merge: &Merge<F>, j_star: usize) -> Result<Self, SumcheckError> {
        let fits = merge
            .polynomials
            .iter()
            .all(|p| j_star <= merge.n_max - p.vars + p.free_vars());
        match fits {
            true => Ok(Self {
                j_star,
                classes: Vec::new(),
                kept: vec![false; merge.claims.len()],
            }),
            false => Err(SumcheckError::Merge {
                rule: "table rounds end before any polynomial's free variables do",
            }),
        }
    }

    /// Group `h`'s tables, from its blocks while they are resident: `block(i, s)` is polynomial
    /// `i` with its group variables set to `s`, `2^{m_i}` entries over its free variables in
    /// increasing position, for every claim of the group.
    ///
    /// # Errors
    ///
    /// [`SumcheckError::Merge`] for a group the merge does not have, a block missing or of
    /// another length than `2^{m_i}`, or a table that cannot be allocated.
    #[allow(clippy::disallowed_methods)]
    pub fn add_group<'b>(
        &mut self,
        merge: &Merge<F>,
        group: usize,
        mut block: impl FnMut(usize, u64) -> Option<&'b Secret<Vec<F::Base>>>,
    ) -> Result<(), SumcheckError> {
        let bad = |rule| SumcheckError::Merge { rule };
        let claims = merge
            .claims
            .get(group)
            .ok_or(bad("the group is one of the merge's"))?;
        if self.kept.get(group) != Some(&false) {
            return Err(bad("each group's tables are kept once"));
        }
        let gamma = merge.gammas[group];
        // Classes: one shape, one point, one set of bits, in the order of their first claim.
        let mut keys: Vec<(&MergePolynomial, &[F], u64)> = Vec::new();
        let mut combined: Vec<SecretBuf<F>> = Vec::new();
        let mut weight = F::ONE;
        for claim in claims {
            let p = &merge.polynomials[claim.polynomial];
            let key = (p, claim.point.as_slice(), claim.group_bits);
            let at = match keys.iter().position(|k| *k == key) {
                Some(at) => at,
                None => {
                    keys.push(key);
                    combined.push(SecretBuf::filled(F::ZERO, 1 << p.free_vars()));
                    keys.len() - 1
                }
            };
            let values = block(claim.polynomial, claim.group_bits)
                .ok_or(bad("every claim's block is given"))?
                .expose_secret();
            if values.len() != 1 << p.free_vars() {
                return Err(bad("a block has 2^m_i entries"));
            }
            for (q, &v) in combined[at].iter_mut().zip(values) {
                *q += weight.mul_by_base(v);
            }
            weight *= gamma;
        }
        for ((p, point, _), mut q) in keys.into_iter().zip(combined) {
            let padding = merge.n_max - p.vars;
            let rounds = self.j_star.saturating_sub(padding).min(p.free_vars());
            // T_t for t < rounds: Q folded at z from its highest variable down.
            let mut tables: Vec<SecretBuf<F>> = Vec::with_capacity(rounds);
            for t in (0..p.free_vars()).rev() {
                if t < rounds {
                    let mut kept = SecretBuf::with_capacity(q.len());
                    kept.extend(q.iter().copied());
                    tables.push(kept);
                }
                let half = q.len() / 2;
                let z = point[t];
                let (low, high) = q.split_at_mut(half);
                for (l, &h) in low.iter_mut().zip(high.iter()) {
                    *l = *l + z * (h - *l);
                }
                q.truncate(half);
            }
            tables.reverse();
            self.classes.push(Class {
                group,
                point: point.to_vec(),
                padding,
                tables,
            });
        }
        self.kept[group] = true;
        Ok(())
    }
}

/// `p(0)` and `p(2)` of a degree-2 round, in place.
type Message<F> = [F; 2];

/// RFC-0054's merge in two passes: the first `j*` rounds from `tables`, kept in the groups' pass,
/// then one pass that regenerates every polynomial, `regenerate(i)` in its natural order, and
/// folds the variables those rounds bound, and the remaining rounds in memory. It sends what
/// [`prove_merge`] sends, when every claim is true.
///
/// # Errors
///
/// [`SumcheckError::Merge`] for tables kept for another merge or missing a group, or a
/// regenerated table of another length than `2^{n_i}`, before the transcript moves for the
/// first; the regenerated tables are checked in the merge pass.
#[allow(clippy::disallowed_methods)]
pub fn prove_merge_two_pass<F, T>(
    transcript: &mut T,
    merge: &Merge<F>,
    tables: &MergeTables<F>,
    mut regenerate: impl FnMut(usize) -> Secret<Vec<F::Base>>,
) -> Result<ProvenBatch<F>, SumcheckError>
where
    F: PolyField,
    T: SumcheckTranscript<F> + ?Sized,
{
    if tables.kept.len() != merge.claims.len() || tables.kept.iter().any(|&k| !k) {
        return Err(SumcheckError::Merge {
            rule: "every group's tables are kept",
        });
    }
    let n_max = merge.n_max;
    let mut labels = Labels::new(Prefix::Merge);
    let mut beta_h = vec![F::ONE; merge.claims.len()];
    for h in 1..beta_h.len() {
        beta_h[h] = beta_h[h - 1] * merge.beta;
    }
    let sums = merge.claimed_sums();
    let mut point: Vec<F> = Vec::with_capacity(n_max);
    let mut rounds = Vec::with_capacity(n_max);
    let mut absorb = |transcript: &mut T, j: usize, message: Message<F>, point: &mut Vec<F>| {
        transcript.absorb_message(labels.round(j), &message);
        let r = transcript.challenge(labels.challenge(j));
        point.push(r);
        rounds.push(message.to_vec());
    };

    // The rounds from the tables: padding from the claims, own rounds from the classes.
    for j in 0..tables.j_star.min(n_max) {
        let mut message: Message<F> = [F::ZERO; 2];
        for (p, &sum) in merge.polynomials.iter().zip(&sums) {
            let padding = n_max - p.vars;
            if j < padding {
                let pad = sum * pow2::<F>(padding - j - 1);
                message[0] += pad;
                message[1] += pad;
            }
        }
        for class in &tables.classes {
            let Some(t) = j.checked_sub(class.padding) else {
                continue;
            };
            let Some(table) = class.tables.get(t) else {
                continue;
            };
            let own = &point[class.padding..j];
            let eq_low = eq_table(own).map_err(|_| SumcheckError::Merge {
                rule: "a round's eq table fits in memory",
            })?;
            let half = 1 << t;
            let (mut l0, mut l1) = (F::ZERO, F::ZERO);
            for (a, &e) in eq_low.iter().enumerate() {
                l0 += e * table[a];
                l1 += e * table[a + half];
            }
            let l2 = l1.double() - l0;
            let z = class.point[t];
            let prefix = eq(&class.point[..t], own).unwrap_or(F::ZERO) * beta_h[class.group];
            message[0] += prefix * (F::ONE - z) * l0;
            message[1] += prefix * (z + z + z - F::ONE) * l2;
        }
        absorb(transcript, j, message, &mut point);
    }

    // The merge pass: every polynomial regenerated, and its variables bound so far folded.
    let j_star = point.len();
    let mut state: Vec<(SecretBuf<F>, Vec<F>, usize)> = Vec::with_capacity(merge.polynomials.len());
    for (i, p) in merge.polynomials.iter().enumerate() {
        let table = regenerate(i);
        if table.expose_secret().len() != 1 << p.vars {
            return Err(SumcheckError::Merge {
                rule: "a regenerated table has 2^n_i entries",
            });
        }
        let ordered = permuted(table.expose_secret(), &p.order());
        let mut values = SecretBuf::with_capacity(ordered.len());
        values.extend(ordered.iter().map(|&b| F::from(b)));
        let mut weights = merge.weights(i)?;
        let padding = n_max - p.vars;
        for &r in point.iter().skip(padding) {
            fold_lowest(&mut values, r);
            fold_lowest(&mut weights, r);
        }
        state.push((values, weights, padding));
    }
    // Each polynomial's sum, for its padding rounds, from its tables as the engine computes it.
    let sums: Vec<F> = state
        .iter()
        .map(|(values, weights, _)| {
            values
                .iter()
                .zip(weights)
                .fold(F::ZERO, |acc, (&v, &w)| acc + v * w)
        })
        .collect();
    for j in j_star..n_max {
        let mut message: Message<F> = [F::ZERO; 2];
        for ((values, weights, padding), &sum) in state.iter().zip(&sums) {
            if j < *padding {
                let pad = sum * pow2::<F>(*padding - j - 1);
                message[0] += pad;
                message[1] += pad;
                continue;
            }
            for (v, w) in values.chunks_exact(2).zip(weights.chunks_exact(2)) {
                message[0] += v[0] * w[0];
                message[1] += (v[1].double() - v[0]) * (w[1].double() - w[0]);
            }
        }
        absorb(transcript, j, message, &mut point);
        let r = point[j];
        for (values, weights, padding) in &mut state {
            if j >= *padding {
                fold_lowest(values, r);
                fold_lowest(weights, r);
            }
        }
    }
    let final_evals: Vec<F> = state.iter().map(|(values, _, _)| values[0]).collect();
    transcript.absorb_message(labels.evals(), &final_evals);
    let vars = merge.polynomials.iter().map(|p| p.vars).collect();
    let evals = (0..final_evals.len()).map(|i| i..i + 1).collect();
    Ok(ProvenBatch::from_parts(
        SumcheckProof {
            rounds,
            final_evals,
        },
        F::ONE,
        point,
        vars,
        evals,
    ))
}

/// Binds the lowest variable to `r`, halving the table. Ascending `k` is safe in place: entry
/// `k` reads entries `2k` and `2k + 1`, which no earlier step wrote.
fn fold_lowest<F: PolyField>(table: &mut impl AsMutTable<F>, r: F) {
    let t = table.entries();
    let half = t.len() / 2;
    for k in 0..half {
        let (a, b) = (t[2 * k], t[2 * k + 1]);
        t[k] = a + r * (b - a);
    }
    table.halve();
}

/// A table [`fold_lowest`] binds in place: a cleared buffer of secret-derived values, or the
/// public weights.
trait AsMutTable<F> {
    fn entries(&mut self) -> &mut [F];
    fn halve(&mut self);
}

impl<F: Copy> AsMutTable<F> for SecretBuf<F> {
    fn entries(&mut self) -> &mut [F] {
        self
    }
    fn halve(&mut self) {
        let half = self.len() / 2;
        self.truncate(half);
    }
}

impl<F: Copy> AsMutTable<F> for Vec<F> {
    fn entries(&mut self) -> &mut [F] {
        self
    }
    fn halve(&mut self) {
        let half = self.len() / 2;
        self.truncate(half);
    }
}
