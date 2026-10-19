// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `SoftmaxRow`'s reciprocal bracket, RFC-0009 relation 3 and Lemma 6, over Goldilocks and
//! `F65537` ([#376](https://github.com/AbdelStark/tessera/issues/376)). The gadget is #122's, so
//! relation 3 is assembled here from what exists: `R_q`'s range check, one lookup into
//! `Range { σ + 1 }`, and two `NonNeg { σ + 18, L }` of `tessera-gadgets` on the slacks of a real
//! row (`ψ_q = 1`), `a_q = 2^{2σ} − R_q·s_q` and `b_q = s_q − 1 − a_q`, computed in the field.
//!
//! - **The wrap.** At `σ = 24` and `s_q = 33,280·2^24`, the sum of 33,280 positions at the row's
//!   maximal score, the true reciprocal `R = 504` and `R' = 33,038,714` both pass every lookup.
//!   `R'·s_q` exceeds `2^{48}` by `P` less a slack below `s_q`: the product wraps once.
//! - **The bound.** At each row sum below, an exhaustive scan of `R ∈ [0, 2^{σ+1})` finds exactly
//!   the reciprocals the proof predicts: the true one, and a second, `⌊(2^{2σ} + P)/s_q⌋`, exactly
//!   when `s_q·2^{σ+1} > 2^{2σ} + P`. Lemma 6's precondition `(2·max_context − 1)·2^{2σ} < P`
//!   keeps every row sum, at most `max_context·2^σ`, at or below that bound: at `σ = 23`, the
//!   largest `max_context = 2^17` admits, its largest row sum has one reciprocal. At `σ = 24`,
//!   `2^{15}` positions have one and `2^{15} + 1` two, as do the last row sum within the bound and
//!   the next.
//! - **Exactness.** Over `F65537`, for every `σ` with `2^{2σ} < p`, every context `M` up to 32,
//!   every row sum up to `M·2^σ` and every `R`: some row sum admits a second reciprocal exactly
//!   when `(2M − 1)·2^{2σ} ≥ p`.
//!
//! Each reciprocal a scan finds passes the assembled lookups, and its neighbours `R ± 1` fail at
//! a slack's `NonNeg`.

mod common;

use common::{Fp, in_table, rejected};
use tessera_field::{F65537, PrimeField};
use tessera_gadgets::{Committed, Constraints, NonNeg, Term};
use tessera_poly::PolyField;
use tessera_tables::{GeneratedTable, Table, TableId, TableKind, TableSpec};

/// `GadgetPolicy.limb_bits`, the limbs' width in production (RFC-0007).
const LIMB_BITS: u8 = 16;

/// `NonNeg`'s width in relation 3: `σ + 18` (RFC-0009 'SoftmaxRow').
fn production_width(sigma: u32) -> u32 {
    sigma + 18
}

/// `⌈log2 m⌉` for `m ≥ 1`.
fn ceil_log2(m: u64) -> u32 {
    u64::BITS - (m - 1).leading_zeros()
}

/// Relation 3 of `SoftmaxRow` on a real row at `σ`: `R_q ∈ Range { σ + 1 }`, then `NonNeg { k, L }`
/// on `a_q` and on `b_q`.
struct Bracket {
    sigma: u32,
    bits: u32,
    limb_bits: u32,
    reciprocal: GeneratedTable,
    slack: NonNeg,
}

impl Bracket {
    fn new(sigma: u32, bits: u32, limb_bits: u8) -> Self {
        let range = |id, bits: u32| {
            let bits = u8::try_from(bits).expect("a small width");
            GeneratedTable::new(TableId(id), TableKind::Range { bits }).expect("a Range table")
        };
        let l = u32::from(limb_bits);
        let top = bits - (bits.div_ceil(l) - 1) * l;
        let specs: Vec<TableSpec> = [range(1, l), range(2, top)]
            .iter()
            .map(|t| t.spec().clone())
            .collect();
        let k = u8::try_from(bits).expect("a small width");
        Self {
            sigma,
            bits,
            limb_bits: l,
            reciprocal: range(0, sigma + 1),
            slack: NonNeg::new(k, limb_bits, &specs).expect("NonNeg"),
        }
    }

    /// The limbs a prover commits for the slack `v`: the digits of its canonical value below
    /// `2^k`. RFC-0009 Lemma 3 makes them the only limbs that can pass.
    fn limbs<F: PrimeField>(&self, v: F) -> Vec<F> {
        let low = v.to_canonical_u64() & ((1 << self.bits) - 1);
        let mask = (1 << self.limb_bits) - 1;
        (0..self.slack.limbs() - 1)
            .map(|j| F::from_u64_reduced((low >> (j * self.limb_bits)) & mask))
            .collect()
    }

    /// The lookups relation 3 rejects for the row sum `s` and the reciprocal `r`: 0 for `R_q`'s
    /// range check, then `a_q`'s `NonNeg` lookups from 1, then `b_q`'s.
    fn rejected<F: PrimeField + PolyField<Base = F>>(&self, s: u64, r: u64) -> Vec<usize> {
        let (s, r) = (F::from_u64_reduced(s), F::from_u64_reduced(r));
        let mut out = Vec::new();
        if !in_table(&self.reciprocal, r, None) {
            out.push(0);
        }
        let a = F::from_u64_reduced(1 << (2 * self.sigma)) - r * s;
        let b = s - F::ONE - a;
        let constraints: Constraints<F> = self.slack.constraints();
        let tables = self.slack.tables();
        let per_slack = constraints.lookups.len();
        for (slack, v) in [a, b].into_iter().enumerate() {
            let limbs = self.limbs(v);
            let value = |term| match term {
                Term::Input => v,
                Term::Committed(Committed::Limb(j)) => limbs[usize::from(j)],
                other => panic!("NonNeg reads no {other:?}"),
            };
            let failed = rejected(&constraints, &tables, value);
            out.extend(failed.into_iter().map(|j| 1 + slack * per_slack + j));
        }
        out
    }
}

/// Every `R ∈ [0, 2^{σ+1})` whose slacks both lie in `[0, 2^k)`, Lemma 3's condition for the two
/// `NonNeg`: an exhaustive scan that steps `a_q` down by `s` in the field.
fn reciprocals<F: PrimeField>(sigma: u32, bits: u32, s: u64) -> Vec<u64> {
    let s = F::from_u64_reduced(s);
    let below = |v: F| v.to_canonical_u64() >> bits == 0;
    let mut a = F::from_u64_reduced(1 << (2 * sigma));
    let mut found = Vec::new();
    for r in 0..1u64 << (sigma + 1) {
        if below(a) && below(s - F::ONE - a) {
            found.push(r);
        }
        a -= s;
    }
    found
}

/// The reciprocals the proof of Lemma 6 predicts for the row sum `s`: the true one,
/// `⌊2^{2σ}/s⌋`, and `⌊(2^{2σ} + P)/s⌋` when `s·2^{σ+1} > 2^{2σ} + P`.
fn predicted(sigma: u32, s: u64, p: u64) -> Vec<u64> {
    let (power, s, p) = (1u128 << (2 * sigma), u128::from(s), u128::from(p));
    let mut out = vec![power / s];
    if s << (sigma + 1) > power + p {
        out.push((power + p) / s);
    }
    out.into_iter()
        .map(|r| u64::try_from(r).expect("below 2^{σ+1}"))
        .collect()
}

/// Scans every reciprocal for the row sum `s` over Goldilocks at the production width, checks the
/// scan against the prediction, each reciprocal found against the assembled lookups, and returns
/// them.
fn goldilocks(sigma: u32, s: u64) -> Vec<u64> {
    let bits = production_width(sigma);
    let found = reciprocals::<Fp>(sigma, bits, s);
    assert_eq!(
        found,
        predicted(sigma, s, Fp::MODULUS),
        "σ = {sigma}, s = {s}"
    );
    let bracket = Bracket::new(sigma, bits, LIMB_BITS);
    for &r in &found {
        assert_eq!(
            bracket.rejected::<Fp>(s, r),
            [0usize; 0],
            "σ = {sigma}, R = {r}"
        );
        // Its neighbours inside the range check fail at a slack.
        let inside = |near: &u64| *near < 1 << (sigma + 1);
        for near in [r.checked_sub(1), r.checked_add(1)]
            .into_iter()
            .flatten()
            .filter(inside)
        {
            let failed = bracket.rejected::<Fp>(s, near);
            assert!(
                !failed.is_empty() && !failed.contains(&0),
                "R = {near}: {failed:?}"
            );
        }
    }
    found
}

#[test]
fn two_reciprocals_pass_relation_3_at_sigma_24() {
    let s: u64 = 33_280 << 24;
    // The true reciprocal: 504·s ≤ 2^48 < 505·s.
    let (power, wide) = (1u128 << 48, u128::from(s));
    assert!(504 * wide <= power && power < 505 * wide);
    // The second: R'·s = 2^48 + P − a' with 0 ≤ a' < s, so the field's a_q is a' and b_q is
    // s − 1 − a', both below 2^42.
    let second: u64 = 33_038_714;
    assert!(second < 1 << 25);
    let a = power + u128::from(Fp::MODULUS) - u128::from(second) * wide;
    assert!(a < wide);
    let bracket = Bracket::new(24, production_width(24), LIMB_BITS);
    for r in [504, second] {
        assert_eq!(bracket.rejected::<Fp>(s, r), [0usize; 0], "R = {r}");
    }
    // The slacks a prover commits for R': a' in limbs, and s − 1 − a'.
    let field =
        Fp::from_u64_reduced(1 << 48) - Fp::from_u64_reduced(second) * Fp::from_u64_reduced(s);
    assert_eq!(u128::from(field.to_canonical_u64()), a);
    // No other reciprocal passes: the scan finds exactly these two.
    assert_eq!(goldilocks(24, s), [504, second]);
}

#[test]
fn below_lemma_6s_bound_the_true_reciprocal_alone_passes() {
    // The largest row sum of a context, max_context positions at the maximal score: 2^17 at
    // σ = 23, the largest σ that context admits, and 2^15 at σ = 24.
    assert_eq!(goldilocks(23, 1 << (17 + 23)), [1 << 6]);
    assert_eq!(goldilocks(24, 1 << (15 + 24)), [1 << 9]);
    // One position more breaks the bound at σ = 24: (2·(2^15 + 1) − 1)·2^48 > P.
    assert_eq!(goldilocks(24, ((1 << 15) + 1) << 24), [511, 33_553_920]);
    // The row sums on either side of s·2^25 = 2^48 + P.
    let p = u128::from(Fp::MODULUS);
    let last = u64::try_from(((1u128 << 48) + p) >> 25).expect("a row sum");
    assert_eq!(goldilocks(24, last).len(), 1);
    assert_eq!(goldilocks(24, last + 1).len(), 2);
}

#[test]
fn over_f65537_lemma_6s_bound_is_exact() {
    let p = F65537::MODULUS;
    let mut sides = [0usize; 2];
    for sigma in (0..).take_while(|&sigma| 1u64 << (2 * sigma) < p) {
        for m in 1..=32u64 {
            // The bracket's width, as σ + 18 is at 2^17 positions: every slack is below
            // m·2^σ ≤ 2^k, and 2^{k+1} ≤ p keeps b_q's comparison exact (Lemma 0's role).
            let bits = sigma + ceil_log2(m) + 1;
            if 1u64 << (bits + 1) > p {
                continue;
            }
            let bracket = Bracket::new(sigma, bits, 4);
            let second = ((1u64 << sigma)..=(m << sigma)).any(|s| {
                let found = reciprocals::<F65537>(sigma, bits, s);
                assert_eq!(
                    found,
                    predicted(sigma, s, p),
                    "σ = {sigma}, M = {m}, s = {s}"
                );
                for &r in &found {
                    assert_eq!(bracket.rejected::<F65537>(s, r), [0usize; 0]);
                }
                found.len() > 1
            });
            let bound = ((2 * m - 1) << (2 * sigma)) < p;
            assert_eq!(second, !bound, "σ = {sigma}, M = {m}");
            sides[usize::from(second)] += 1;
        }
    }
    // Both sides of the bound occur.
    assert!(sides.iter().all(|&count| count > 0), "{sides:?}");
}
