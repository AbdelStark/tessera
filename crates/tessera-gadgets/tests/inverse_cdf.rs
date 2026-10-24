// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0009 Lemma 10 in RFC-0058's columns: honest rows, neighbouring tokens, the edges and the
//! counts. Assignments are built here from RFC-0058's formulas, not by the gadget, and each
//! obligation is evaluated on them: the equalities pointwise, the range requests against their
//! tables' widths.

mod common;
use common::{Fp, Fp2, Session, SplitMix64, TvFp2, mle_fp, tv_fp2, tv_fp2s};
use tessera_field::{F65537, Field, PrimeField};
use tessera_gadgets::{
    CdfColumn as Col, CdfTerm as T, CompareColumn as C, Edge, GadgetError, InverseCdf, OneHot, Side,
};
use tessera_sumcheck::{Elementwise, Prefix, prove_batch};
use tessera_tables::{GeneratedTable, Table, TableId, TableKind, TableSpec};
use tv_core::sumcheck as tv;

fn specs() -> Vec<TableSpec> {
    (1..=32)
        .map(|bits| {
            GeneratedTable::new(TableId(u32::from(bits)), TableKind::Range { bits })
                .unwrap()
                .spec()
                .clone()
        })
        .collect()
}

fn gadget(vocab: usize, sigma: u8) -> InverseCdf<Fp> {
    InverseCdf::new(vocab, sigma, &specs()).unwrap()
}

fn padded(weights: &[u64]) -> Vec<Fp> {
    let mut out: Vec<Fp> = weights.iter().map(|&w| Fp::from_u64_reduced(w)).collect();
    out.resize(weights.len().next_power_of_two(), Fp::ZERO);
    out
}

/// RFC-0009 Lemma 10's token, by a scan in exact integers.
fn reference(weights: &[u64], u: u64) -> Option<usize> {
    let total: u128 = weights.iter().map(|&w| u128::from(w)).sum();
    let mut prefix = 0u128;
    for (v, &w) in weights.iter().enumerate() {
        prefix += u128::from(w);
        if prefix << 32 > u128::from(u) * total {
            return Some(v);
        }
    }
    None
}

/// The row's sums for a token, RFC-0058 'The vocabulary sums': `Z`, `B = C(ŷ − 1)`, `S = e_ŷ`.
fn sums(weights: &[u64], token: Option<usize>) -> (u128, u128, u128) {
    let total = weights.iter().map(|&w| u128::from(w)).sum();
    match token {
        Some(t) => (
            total,
            weights[..t].iter().map(|&w| u128::from(w)).sum(),
            u128::from(weights[t]),
        ),
        None => (total, 0, 0),
    }
}

struct Row {
    values: Vec<(Col, Fp)>,
    u: u64,
    sums: (u128, u128, u128),
    real: bool,
}

/// A prover's assignment from RFC-0058's formulas, honest or not: the limbs of `u`, `Z` and
/// `B`, each comparison's coefficients, their normalization, and the modular subtraction with
/// its borrows. A false comparison passes every check but its final borrow's.
fn assignment(width: u8, u: u64, (z, b, s): (u128, u128, u128), real: bool) -> Row {
    let k = usize::from(width).div_ceil(16);
    let m = k + 2;
    let limbs = |x: u128| -> Vec<u128> {
        (0..k)
            .map(|j| {
                if j + 1 == k {
                    x >> (16 * j)
                } else {
                    (x >> (16 * j)) & 0xffff
                }
            })
            .collect()
    };
    let draw = [u128::from(u) & 0xffff, u128::from(u) >> 16];
    let (zl, bl) = (limbs(z), limbs(b));
    let mut values = vec![(Col::DrawLimb, Fp::from_u64_reduced(draw[0] as u64))];
    for (j, &limb) in zl[..k - 1].iter().enumerate() {
        values.push((Col::TotalLimb(j as u8), Fp::from_u64_reduced(limb as u64)));
    }
    for (j, &limb) in bl[..k - 1].iter().enumerate() {
        values.push((Col::BelowLimb(j as u8), Fp::from_u64_reduced(limb as u64)));
    }
    let product: Vec<u128> = (0..m)
        .map(|j| {
            (0..2)
                .flat_map(|i| (0..k).map(move |h| (i, h)))
                .filter(|&(i, h)| i + h == j)
                .map(|(i, h)| draw[i] * zl[h])
                .sum()
        })
        .collect();
    let shifted: Vec<u128> = (0..m).map(|j| if j >= 2 { bl[j - 2] } else { 0 }).collect();
    for edge in [Edge::Lower, Edge::Upper] {
        let (x, y): (Vec<u128>, Vec<u128>) = match edge {
            Edge::Lower => (shifted.clone(), product.clone()),
            Edge::Upper => (
                (0..m)
                    .map(|j| product[j] + u128::from(j == 0 && real))
                    .collect(),
                (0..m)
                    .map(|j| shifted[j] + if j == 2 { s } else { 0 })
                    .collect(),
            ),
        };
        let mut digits = [vec![0i128; m], vec![0i128; m]];
        let mut arrays: Vec<(Col, Fp)> = Vec::new();
        for (index, (side, input)) in [(Side::Left, x), (Side::Right, y)].into_iter().enumerate() {
            let mut carry = 0u128;
            let mut carries = Vec::new();
            for (j, a) in input.into_iter().enumerate() {
                let sum = a + carry;
                digits[index][j] = (sum & 0xffff) as i128;
                carry = sum >> 16;
                arrays.push((
                    Col::Compare(edge, C::Digit(side, j)),
                    Fp::from_u64_reduced((sum & 0xffff) as u64),
                ));
                carries.push((
                    Col::Compare(edge, C::Carry(side, j)),
                    Fp::from_u64_reduced(carry as u64),
                ));
            }
            arrays.extend(carries);
        }
        let mut differences = Vec::new();
        let mut borrows = Vec::new();
        let mut borrow = 0;
        for (j, (&right, &left)) in digits[1].iter().zip(&digits[0]).enumerate() {
            let delta = right - left - borrow;
            borrow = i128::from(delta < 0);
            differences.push((
                Col::Compare(edge, C::Difference(j)),
                Fp::from_i64(delta.rem_euclid(1 << 16) as i64),
            ));
            borrows.push((
                Col::Compare(edge, C::Borrow(j)),
                Fp::from_i64(borrow as i64),
            ));
        }
        values.extend(arrays);
        values.extend(differences);
        values.extend(borrows);
    }
    Row {
        values,
        u,
        sums: (z, b, s),
        real,
    }
}

/// The indices of the equalities that are not zero and of the range requests that miss their
/// tables, on `row`.
fn failures(g: &InverseCdf<Fp>, row: &Row) -> (Vec<usize>, Vec<usize>) {
    let c = g.constraints::<Fp>().unwrap();
    let field = |x: u128| Fp::from_u64_reduced(x as u64);
    let value = |t| match t {
        T::Draw => Fp::from_u64_reduced(row.u),
        T::Total => field(row.sums.0),
        T::Below => field(row.sums.1),
        T::Selected => field(row.sums.2),
        T::RowSelector => Fp::from_u64_reduced(u64::from(row.real)),
        T::Column(column) => {
            row.values
                .iter()
                .find(|(k, _)| *k == column)
                .unwrap_or_else(|| panic!("{column:?} is assigned"))
                .1
        }
    };
    let equalities = c
        .equalities
        .iter()
        .enumerate()
        .filter(|(_, e)| e.evaluate(value) != Fp::ZERO)
        .map(|(i, _)| i)
        .collect();
    let lookups = c
        .lookups
        .iter()
        .enumerate()
        .filter(|(_, l)| {
            let TableKind::Range { bits } = specs()[l.table.0 as usize - 1].kind else {
                panic!("a range table")
            };
            l.key.evaluate(value).to_canonical_u64() >= 1u64 << bits
        })
        .map(|(i, _)| i)
        .collect();
    (equalities, lookups)
}

/// The final borrow's equality of each comparison: each has `2(m + 1)` normalization
/// equalities, then `m + 1` for the subtraction (RFC-0050).
fn final_borrow(g: &InverseCdf<Fp>, edge: Edge) -> usize {
    let per = 3 * (g.limbs() + 2) + 3;
    per * (1 + usize::from(edge == Edge::Upper)) - 1
}

fn random_weights(rng: &mut SplitMix64, vocab: usize, sigma: u8) -> Vec<u64> {
    let zeros = rng.next_u64() % 4;
    let mut weights: Vec<u64> = (0..vocab)
        .map(|_| {
            let w = rng.next_u64() % ((1 << sigma) + 1);
            if rng.next_u64() % 4 < zeros { 0 } else { w }
        })
        .collect();
    // The vocabulary certificate's maximal logit has key 0, whose exponential is 2^σ.
    let top = (rng.next_u64() % vocab as u64) as usize;
    weights[top] = 1 << sigma;
    weights
}

#[test]
fn honest_rows_choose_the_reference_token_and_pass_every_obligation() {
    let mut rng = SplitMix64::new(0x0144_0001);
    for case in 0..400 {
        let vocab = 1 + (rng.next_u64() % 40) as usize;
        let sigma = (rng.next_u64() % 33) as u8;
        let g = gadget(vocab, sigma);
        let weights = random_weights(&mut rng, vocab, sigma);
        let u = match case % 5 {
            0 => 0,
            1 => u32::MAX.into(),
            _ => rng.next_u64() >> 32,
        };
        let expected = reference(&weights, u).unwrap();
        let e = padded(&weights);
        let h = g.sample(&e, Fp::from_u64_reduced(u), true).unwrap();
        let one = OneHot::<Fp>::new(vocab)
            .unwrap()
            .witness(expected as i64, true)
            .unwrap();
        assert_eq!(
            h.indicator(),
            one.indicator(),
            "case {case}: the reference token"
        );
        let witness = g.witness(&e, Fp::from_u64_reduced(u), &h, true).unwrap();
        let row = assignment(g.width(), u, sums(&weights, Some(expected)), true);
        assert_eq!(failures(&g, &row), (vec![], vec![]), "case {case}");
        let constraints = g.constraints::<Fp>().unwrap();
        assert_eq!(witness.columns(), constraints.committed, "RFC-0058 order");
        for (column, value) in &row.values {
            assert_eq!(
                witness.value(*column).map(Fp::from_i64),
                Some(*value),
                "case {case}: {column:?}"
            );
        }
        let evaluated = g.evaluate(&e, &h).unwrap();
        let (z, b, s) = sums(&weights, Some(expected));
        assert_eq!(
            &evaluated.sums[..],
            [z, b, s].map(|x| Fp::from_u64_reduced(x as u64))
        );
    }
}

#[test]
fn neighbouring_tokens_fail_exactly_their_comparisons_final_borrow() {
    let mut rng = SplitMix64::new(0x0144_0002);
    let mut seen = [0; 2];
    for _ in 0..400 {
        let vocab = 2 + (rng.next_u64() % 30) as usize;
        let sigma = (rng.next_u64() % 33) as u8;
        let g = gadget(vocab, sigma);
        let weights = random_weights(&mut rng, vocab, sigma);
        let u = rng.next_u64() >> 32;
        let token = reference(&weights, u).unwrap();
        let e = padded(&weights);
        let one = OneHot::<Fp>::new(vocab).unwrap();
        // `ŷ + 1` moves the lower edge to C(ŷ), past the draw; `ŷ − 1` moves the upper edge to
        // C(ŷ − 1), at or below it.
        for (neighbour, edge) in [
            (token.checked_add(1).filter(|&t| t < vocab), Edge::Lower),
            (token.checked_sub(1), Edge::Upper),
        ] {
            let Some(t) = neighbour else { continue };
            let row = assignment(g.width(), u, sums(&weights, Some(t)), true);
            let (equalities, lookups) = failures(&g, &row);
            assert!(
                lookups.is_empty(),
                "every limb and borrow stays in its table"
            );
            // A zero weight between them can make both edges fail.
            assert!(
                equalities.contains(&final_borrow(&g, edge)),
                "{edge:?}: {equalities:?}"
            );
            assert!(
                equalities
                    .iter()
                    .all(|&i| i == final_borrow(&g, Edge::Lower)
                        || i == final_borrow(&g, Edge::Upper))
            );
            let h = one.witness(t as i64, true).unwrap();
            assert_eq!(
                g.witness(&e, Fp::from_u64_reduced(u), &h, true).map(|_| ()),
                Err(InverseCdf::<Fp>::MISMATCH)
            );
            seen[usize::from(edge == Edge::Upper)] += 1;
        }
    }
    assert!(seen.iter().all(|&n| n > 100), "{seen:?}");
}

#[test]
fn a_token_of_weight_zero_is_never_certified() {
    let g = gadget(4, 8);
    let weights = [256, 0, 100, 0];
    for u in [0, 1 << 30, 1 << 31, u32::MAX.into()] {
        for t in [1, 3] {
            let row = assignment(g.width(), u, sums(&weights, Some(t)), true);
            let (equalities, _) = failures(&g, &row);
            assert!(!equalities.is_empty(), "u {u}, token {t}");
        }
        assert_ne!(reference(&weights, u), Some(1));
        assert_ne!(reference(&weights, u), Some(3));
    }
}

#[test]
fn a_draw_of_32_bits_or_more_fails_its_top_limb() {
    let g = gadget(3, 4);
    let weights = [16, 5, 9];
    let e = padded(&weights);
    for u in [1u64 << 32, (1 << 40) + 7] {
        let row = assignment(g.width(), u, sums(&weights, Some(0)), true);
        // The draw's requests come first: its low limb, then its virtual top.
        assert!(failures(&g, &row).1.contains(&1), "u {u}");
        assert!(matches!(
            g.sample(&e, Fp::from_u64_reduced(u), true),
            Err(GadgetError::Input { .. })
        ));
    }
    // The field's top values are no draw either.
    assert!(g.sample(&e, -Fp::ONE, true).is_err());
}

#[test]
fn a_real_row_without_mass_has_no_token() {
    let g = gadget(3, 4);
    let e = padded(&[0, 0, 0]);
    assert!(matches!(
        g.sample(&e, Fp::from_u64_reduced(5), true),
        Err(GadgetError::Input { .. })
    ));
    // Every token leaves the upper comparison 1 ≤ 0.
    for t in 0..3 {
        let row = assignment(g.width(), 5, sums(&[0, 0, 0], Some(t)), true);
        assert_eq!(failures(&g, &row).0, vec![final_borrow(&g, Edge::Upper)]);
    }
}

#[test]
fn a_padded_row_has_the_zero_witness_for_any_draw() {
    let g = gadget(5, 16);
    let e = padded(&[0; 5]);
    for u in [0, 12345, u32::MAX.into()] {
        let h = g.sample(&e, Fp::from_u64_reduced(u), false).unwrap();
        assert!(h.indicator().iter().all(|&x| x == Fp::ZERO));
        let witness = g.witness(&e, Fp::from_u64_reduced(u), &h, false).unwrap();
        for (&column, &value) in witness.columns().iter().zip(witness.values()) {
            let expected = if column == Col::DrawLimb {
                (u & 0xffff) as i64
            } else {
                0
            };
            assert_eq!(value, expected, "{column:?}");
        }
        let row = assignment(g.width(), u, (0, 0, 0), false);
        assert_eq!(failures(&g, &row), (vec![], vec![]));
    }
    // A padded row's exponentials are zero, and its indicator is.
    let e = padded(&[0, 3, 0, 0, 0]);
    assert!(g.sample(&e, Fp::ZERO, false).is_err());
}

/// A padded row with mass in its exponentials and a draw of 0 passes both comparisons, with
/// non-zero partition limbs: the padding zero-check alone refuses it (RFC-0058 'Padded rows').
#[test]
fn a_padded_rows_partition_is_pinned_to_zero() {
    let g = gadget(128_256, 32);
    let last = g.constraints::<Fp>().unwrap().equalities.len() - 1;
    for z in [1u128, 1 << 40, 128_256 << 32] {
        let row = assignment(g.width(), 0, (z, 0, 0), false);
        assert_eq!(failures(&g, &row), (vec![last], vec![]), "Z = {z}");
    }
    // On a real row the zero-check is (1 − 1)·Z = 0.
    let row = assignment(g.width(), 0, (1 << 40, 0, 1), true);
    assert!(!failures(&g, &row).0.contains(&last));
}

/// RFC-0058 'Committed values, range requests and counts', written out here: `u_0`, `Z`'s low
/// limbs, `B`'s, then each comparison's six arrays in RFC-0050's order; the requests in the same
/// order, each decomposition's virtual top after its low limbs.
#[test]
fn the_committed_values_and_requests_are_in_rfc_0058_order() {
    for (vocab, sigma) in [(1, 0), (5, 16), (128_256, 16), (128_256, 32), (1 << 18, 32)] {
        let g = gadget(vocab, sigma);
        let k = g.limbs();
        let m = k + 2;
        let mut expected = vec![Col::DrawLimb];
        expected.extend((0..k - 1).map(|j| Col::TotalLimb(j as u8)));
        expected.extend((0..k - 1).map(|j| Col::BelowLimb(j as u8)));
        for edge in [Edge::Lower, Edge::Upper] {
            for side in [Side::Left, Side::Right] {
                expected.extend((0..m).map(|j| Col::Compare(edge, C::Digit(side, j))));
                expected.extend((0..m).map(|j| Col::Compare(edge, C::Carry(side, j))));
            }
            expected.extend((0..m).map(|j| Col::Compare(edge, C::Difference(j))));
            expected.extend((0..m).map(|j| Col::Compare(edge, C::Borrow(j))));
        }
        let c = g.constraints::<Fp>().unwrap();
        assert_eq!(c.committed, expected, "vocab {vocab}, σ {sigma}");
        // Each request's key: a top reads its decomposition's input; every other key is one
        // committed column, in the committed order.
        let mut keys = Vec::new();
        for (low, input) in [(1, T::Draw), (k - 1, T::Total), (k - 1, T::Below)] {
            keys.extend((0..low).map(|_| None));
            keys.push(Some(input));
        }
        let read: Vec<Option<T>> = c
            .lookups
            .iter()
            .take(keys.len())
            .map(|l| {
                l.key
                    .terms
                    .iter()
                    .map(|t| t.1)
                    .find(|t| matches!(t, T::Draw | T::Total | T::Below))
            })
            .collect();
        assert_eq!(read, keys);
        let singles: Vec<Col> = c
            .lookups
            .iter()
            .filter_map(|l| match l.key.terms.as_slice() {
                [(one, T::Column(column))] if *one == Fp::ONE && l.key.constant == Fp::ZERO => {
                    Some(*column)
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            singles, expected,
            "every committed value has its own request, in order"
        );
    }
}

/// Every committed value is pinned: one outside its table fails its own request, and one moved by
/// one inside its table fails an equality (RFC-0009 Lemmas 3 and 4).
#[test]
fn every_committed_value_fails_alone_when_moved() {
    let g = gadget(5, 20);
    let weights = [1 << 20, 77, 0, 1 << 19, 12345];
    let u = 2_900_000_000u64;
    let token = reference(&weights, u).unwrap();
    let honest = assignment(g.width(), u, sums(&weights, Some(token)), true);
    assert_eq!(failures(&g, &honest), (vec![], vec![]));
    let c = g.constraints::<Fp>().unwrap();
    for (i, column) in c.committed.iter().enumerate() {
        let request = c
            .lookups
            .iter()
            .position(|l| l.key.terms == [(Fp::ONE, T::Column(*column))])
            .unwrap();
        let mut row = assignment(g.width(), u, sums(&weights, Some(token)), true);
        row.values[i].1 = -Fp::ONE;
        assert!(
            failures(&g, &row).1.contains(&request),
            "{column:?} outside its table"
        );
        let mut row = assignment(g.width(), u, sums(&weights, Some(token)), true);
        let moved = row.values[i].1 + Fp::ONE;
        row.values[i].1 =
            if moved.to_canonical_u64() == 2 && matches!(column, Col::Compare(_, C::Borrow(_))) {
                Fp::ZERO
            } else {
                moved
            };
        let (equalities, _) = failures(&g, &row);
        assert!(!equalities.is_empty(), "{column:?} moved by one");
    }
}

/// Bounded mode's indicator needs OneHot's obligations: the row's alone admit a non-Boolean
/// indicator and one on the vocabulary's padding, which booleanity and the vocabulary relation
/// refuse (RFC-0058 'Parameters and inputs').
#[test]
fn bounded_mode_proves_the_sampled_indicator_one_hot() {
    let z = [3u64, 5, 7].map(|x| Fp2::from(Fp::from_u64_reduced(x)));
    let tv_z: Vec<TvFp2> = z.iter().map(|&x| tv_fp2(x)).collect();
    let column = |xs: &[i64]| {
        tessera_gadgets::OneHotWitness::from_column(xs.iter().map(|&x| Fp::from_i64(x)).collect())
    };
    // Non-Boolean: ĥ = (0, 1, 1, −1) has count 1 and gives B = 7, S = 11 beside the honest
    // B = 16, S = 5, and the row's obligations hold for both.
    let g = gadget(4, 4);
    let weights = [16, 5, 9, 3];
    let u = 2_082_408_387u64;
    assert_eq!(reference(&weights, u), Some(1));
    let row = assignment(g.width(), u, (33, 7, 11), true);
    assert_eq!(failures(&g, &row), (vec![], vec![]));
    let h = column(&[0, 1, 1, -1]);
    let relations = g.sampled_relations::<Fp2>(&h).unwrap();
    assert!(
        subproof(
            &relations.booleanity,
            &z[..2],
            Fp2::ZERO,
            &[h.indicator()],
            |p, x| { tv::eq(&tv_z[..2], p).unwrap() * x[0] * (TvFp2::ONE - x[0]) }
        )
        .is_err()
    );
    subproof(&relations.count, &[], Fp2::ONE, &[h.indicator()], |_, x| {
        x[0]
    })
    .unwrap();
    // On the padding: vocabulary 5 padded to 8, ĥ at 7, and padded exponentials −13, 0, 1 give
    // Z = 40, B = 27 and S = 1, which pass the row's obligations.
    let g = gadget(5, 4);
    let u = 3_000_000_000u64;
    let row = assignment(g.width(), u, (40, 27, 1), true);
    assert_eq!(failures(&g, &row), (vec![], vec![]));
    let h = column(&[0, 0, 0, 0, 0, 0, 0, 1]);
    let relations = g.sampled_relations::<Fp2>(&h).unwrap();
    let valid: Vec<Fp> = (0..8)
        .map(|v| Fp::from_u64_reduced(u64::from(v < 5)))
        .collect();
    assert!(
        subproof(
            &relations.vocabulary,
            &z,
            Fp2::ZERO,
            &[h.indicator()],
            |p, x| { tv::eq(&tv_z, p).unwrap() * (TvFp2::ONE - mle_fp(&valid, p)) * x[0] }
        )
        .is_err()
    );
    // Only the vocabulary relation refuses it: it is Boolean, with count 1.
    subproof(
        &relations.booleanity,
        &z,
        Fp2::ZERO,
        &[h.indicator()],
        |p, x| tv::eq(&tv_z, p).unwrap() * x[0] * (TvFp2::ONE - x[0]),
    )
    .unwrap();
    subproof(&relations.count, &[], Fp2::ONE, &[h.indicator()], |_, x| {
        x[0]
    })
    .unwrap();
    // The honest indicator passes all three.
    let mut e = padded(&[16, 5, 9, 3, 7]);
    e[5] = -Fp::from_u64_reduced(13);
    e[7] = Fp::ONE;
    let h = g.sample(&e, Fp::from_u64_reduced(u), true).unwrap();
    let relations = g.sampled_relations::<Fp2>(&h).unwrap();
    subproof(
        &relations.booleanity,
        &z,
        Fp2::ZERO,
        &[h.indicator()],
        |p, x| tv::eq(&tv_z, p).unwrap() * x[0] * (TvFp2::ONE - x[0]),
    )
    .unwrap();
    subproof(
        &relations.vocabulary,
        &z,
        Fp2::ZERO,
        &[h.indicator()],
        |p, x| tv::eq(&tv_z, p).unwrap() * (TvFp2::ONE - mle_fp(&valid, p)) * x[0],
    )
    .unwrap();
    subproof(&relations.count, &[], Fp2::ONE, &[h.indicator()], |_, x| {
        x[0]
    })
    .unwrap();
}

#[test]
fn malformed_indicators_are_refused() {
    let g = gadget(3, 4);
    let e = padded(&[16, 5, 9]);
    let u = Fp::from_u64_reduced(1 << 31);
    let column = |xs: [u64; 4]| {
        tessera_gadgets::OneHotWitness::from_column(
            xs.iter().map(|&x| Fp::from_u64_reduced(x)).collect(),
        )
    };
    // The domain's refusal, not the comparisons': each would also fail a comparison.
    let domain = |result: Result<_, GadgetError>| matches!(result, Err(GadgetError::Input { rule }) if rule.contains("count ψ"));
    for bad in [[1, 1, 0, 0], [0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 1]] {
        assert!(
            domain(g.witness(&e, u, &column(bad), true).map(|_| ())),
            "{bad:?}"
        );
    }
    assert!(domain(
        g.witness(&e, u, &column([1, 0, 0, 0]), false).map(|_| ())
    ));
    let short = tessera_gadgets::OneHotWitness::from_column([Fp::ONE].into_iter().collect());
    assert!(matches!(
        g.witness(&e, u, &short, true),
        Err(GadgetError::Shape { .. })
    ));
}

#[test]
fn the_largest_partition_fits_its_limbs() {
    let vocab = 1 << 18;
    let g = gadget(vocab, 32);
    assert_eq!((g.width(), g.limbs()), (51, 4));
    let weights = vec![1u64 << 32; vocab];
    let e = padded(&weights);
    for u in [0, 1, (1u64 << 31) + 12345, u32::MAX.into()] {
        let h = g.sample(&e, Fp::from_u64_reduced(u), true).unwrap();
        let token = reference(&weights, u).unwrap();
        assert_eq!(h.indicator()[token], Fp::ONE);
        // Equal weights: the token is ⌊u·vocab/2^32⌋.
        assert_eq!(token as u64, (u * vocab as u64) >> 32);
        g.witness(&e, Fp::from_u64_reduced(u), &h, true).unwrap();
        let row = assignment(g.width(), u, sums(&weights, Some(token)), true);
        assert_eq!(failures(&g, &row), (vec![], vec![]));
    }
}

#[test]
fn the_counts_and_request_order_follow_rfc_0058_at_every_width() {
    for width in 1..=51u8 {
        let (vocab, sigma) = if width <= 33 {
            (1, width - 1)
        } else {
            (1 << (width - 33), 32)
        };
        let g = gadget(vocab, sigma);
        assert_eq!(g.width(), width);
        let k = usize::from(width).div_ceil(16);
        let c = g.constraints::<Fp>().unwrap();
        assert_eq!(
            c.committed.len(),
            1 + 2 * (k - 1) + 12 * (k + 2),
            "W = {width}"
        );
        assert_eq!(c.lookups.len(), 2 + 2 * k + 12 * (k + 2), "W = {width}");
        assert_eq!(c.equalities.len(), 2 * (3 * (k + 2) + 3) + 1);
        let bits: Vec<u32> = c.lookups.iter().map(|l| l.table.0).collect();
        let top = u32::from(width) - 16 * (k as u32 - 1);
        let mut expected = vec![16, 16];
        for _ in 0..2 {
            expected.extend(std::iter::repeat_n(16, k - 1));
            expected.push(top);
        }
        assert_eq!(bits[..2 + 2 * k], expected, "W = {width}");
    }
    // RFC-0058's table.
    for (vocab, sigma, committed, requests) in [
        (128_256, 16, 65, 68),
        (128_256, 32, 79, 82),
        (1 << 18, 32, 79, 82),
    ] {
        let c = gadget(vocab, sigma).constraints::<Fp>().unwrap();
        assert_eq!((c.committed.len(), c.lookups.len()), (committed, requests));
    }
}

#[test]
fn parameters_outside_the_boxes_are_refused() {
    assert!(matches!(
        InverseCdf::<Fp>::new(4, 33, &specs()),
        Err(GadgetError::Shape { .. })
    ));
    assert!(matches!(
        InverseCdf::<Fp>::new(0, 8, &specs()),
        Err(GadgetError::Shape { .. })
    ));
    assert!(matches!(
        InverseCdf::<Fp>::new((1 << 18) + 1, 8, &specs()),
        Err(GadgetError::Shape { .. })
    ));
    // F65537 holds neither the draw's box nor the comparisons' equations.
    assert!(matches!(
        InverseCdf::<F65537>::new(4, 8, &specs()),
        Err(GadgetError::Shape { .. })
    ));
    assert!(matches!(
        gadget(4, 8).constraints::<F65537>(),
        Err(GadgetError::Shape { .. })
    ));
    // A missing range table.
    let without: Vec<TableSpec> = specs()
        .into_iter()
        .filter(|s| s.kind != TableKind::Range { bits: 16 })
        .collect();
    assert!(matches!(
        InverseCdf::<Fp>::new(4, 8, &without),
        Err(GadgetError::Shape { .. })
    ));
}

fn subproof(
    relation: &Elementwise<'_, Fp2>,
    point: &[Fp2],
    target: Fp2,
    operands: &[&[Fp]],
    mut final_relation: impl FnMut(&[TvFp2], &[TvFp2]) -> TvFp2,
) -> Result<(), tv::SumcheckError> {
    let session = Session::random(&mut SplitMix64::new(0x0144_0003));
    let tables = relation.tables(point).unwrap();
    let mut transcript = session.transcript();
    let degree = relation.degree();
    let proven = prove_batch(
        &mut transcript,
        Prefix::Stage(0),
        degree,
        &[tables.instance()],
    )
    .unwrap();
    let proof = tv::SumcheckProof {
        rounds: proven.proof().rounds.iter().map(|r| tv_fp2s(r)).collect(),
        final_evals: tv_fp2s(&proven.proof().final_evals),
    };
    let mut verifier = session.tv_transcript();
    let checked = tv::verify_batch(
        &mut verifier,
        tv::Prefix::Stage(0),
        degree,
        &[tv::Instance {
            claim: tv_fp2(target),
            vars: relation.shape().vars,
            evals: operands.len(),
        }],
        &proof,
        |_, r, e| Some(final_relation(r, e)),
    )?;
    for (column, &value) in operands.iter().zip(&proof.final_evals) {
        let r = checked
            .point_of(column.len().trailing_zeros() as usize)
            .unwrap();
        assert_eq!(value, mle_fp(column, r), "consume each operand claim");
    }
    Ok(())
}

/// The three contractions prove their sums against the verifier's independent sumcheck, with
/// the vocabulary indicator evaluated by the verifier; a prefix of the wrong token is refused,
/// and so is a partition that counts the vocabulary's padding.
#[test]
fn the_vocabulary_contractions_prove_their_sums() {
    let vocab = 6;
    let g = gadget(vocab, 8);
    let weights = [256, 3, 0, 100, 7, 41];
    // Vocabulary padding is never read unmasked: give it values that would move every sum.
    let mut e = padded(&weights);
    e[6] = Fp::from_u64_reduced(999);
    e[7] = -Fp::ONE;
    let u = 2_000_000_000u64;
    let token = reference(&weights, u).unwrap();
    let h = g.sample(&e, Fp::from_u64_reduced(u), true).unwrap();
    assert_eq!(h.indicator()[token], Fp::ONE);
    g.witness(&e, Fp::from_u64_reduced(u), &h, true).unwrap();
    let evaluated = g.evaluate(&e, &h).unwrap();
    let relations = g.relations::<Fp2>(&e, &h, &evaluated).unwrap();
    let valid: Vec<Fp> = (0..8)
        .map(|v| Fp::from_u64_reduced(u64::from(v < vocab)))
        .collect();
    let (z, b, s) = sums(&weights, Some(token));
    let lift = |x: u128| Fp2::from(Fp::from_u64_reduced(x as u64));
    let valid_at = |r: &[TvFp2]| mle_fp(&valid, r);
    subproof(&relations.total, &[], lift(z), &[&e], |r, x| {
        valid_at(r) * x[0]
    })
    .unwrap();
    assert_eq!(
        &evaluated.sums[..],
        [z, b, s].map(|x| Fp::from_u64_reduced(x as u64))
    );
    subproof(
        &relations.below,
        &[],
        lift(b),
        &[&e, &evaluated.suffix],
        |_, x| x[0] * x[1],
    )
    .unwrap();
    subproof(
        &relations.selected,
        &[],
        lift(s),
        &[h.indicator(), &e],
        |_, x| x[0] * x[1],
    )
    .unwrap();
    // Another token's exponential is refused by the same relation.
    assert!(
        subproof(
            &relations.selected,
            &[],
            lift(u128::from(weights[token + 1])),
            &[h.indicator(), &e],
            |_, x| x[0] * x[1],
        )
        .is_err()
    );
    let unmasked = e.iter().fold(Fp2::ZERO, |acc, &x| acc + Fp2::from(x));
    assert!(
        subproof(&relations.total, &[], unmasked, &[&e], |r, x| valid_at(r)
            * x[0])
        .is_err()
    );
    // The prefix of the next token is refused by the same relation.
    let (_, next, _) = sums(&weights, Some(token + 1));
    assert_ne!(next, b);
    assert!(
        subproof(
            &relations.below,
            &[],
            lift(next),
            &[&e, &evaluated.suffix],
            |_, x| x[0] * x[1]
        )
        .is_err()
    );
    // The suffix's dot product: L̃(ρ) = Σ_v w_v(ρ)·ĥ_v at a point.
    let rho: Vec<Fp2> = [3u64, 5, 11]
        .map(|x| Fp2::from(Fp::from_u64_reduced(x)))
        .to_vec();
    let weights_at = g.suffix_weights(&rho).unwrap();
    let dot = weights_at
        .iter()
        .zip(h.indicator())
        .fold(Fp2::ZERO, |acc, (&w, &x)| acc + w * Fp2::from(x));
    let tv_rho: Vec<TvFp2> = rho.iter().map(|&x| tv_fp2(x)).collect();
    assert_eq!(tv_fp2(dot), mle_fp(&evaluated.suffix, &tv_rho));
}

/// Bounded mode's count relation is load-bearing: a two-hot indicator on a real row, and a one on
/// a padded row, pass every row obligation, booleanity and the vocabulary relation, and only the
/// count refuses them.
#[test]
fn only_the_count_refuses_a_two_hot_or_a_padded_rows_indicator() {
    let z = [
        Fp2::from(Fp::from_u64_reduced(4)),
        Fp2::from(Fp::from_u64_reduced(9)),
    ];
    let tv_z: Vec<TvFp2> = z.iter().map(|&x| tv_fp2(x)).collect();
    let valid: Vec<Fp> = (0..4)
        .map(|v| Fp::from_u64_reduced(u64::from(v < 4)))
        .collect();
    let g = gadget(4, 4);
    let column = |xs: &[i64]| {
        tessera_gadgets::OneHotWitness::from_column(xs.iter().map(|&x| Fp::from_i64(x)).collect())
    };
    for (h, real, u, sums, count) in [
        // Real: weights 16, 5, 9, 3 and u = 3·10^9 sample token 2; (1, 1, 0, 0) gives B = 16,
        // S = 21.
        (
            column(&[1, 1, 0, 0]),
            true,
            3_000_000_000u64,
            (33, 16, 21),
            1u64,
        ),
        // Padded: e = 0, so every sum is 0 whatever the indicator.
        (column(&[0, 1, 0, 0]), false, 77, (0, 0, 0), 0),
    ] {
        let row = assignment(g.width(), u, sums, real);
        assert_eq!(failures(&g, &row), (vec![], vec![]), "real {real}");
        let relations = g.sampled_relations::<Fp2>(&h).unwrap();
        subproof(
            &relations.booleanity,
            &z,
            Fp2::ZERO,
            &[h.indicator()],
            |p, x| tv::eq(&tv_z, p).unwrap() * x[0] * (TvFp2::ONE - x[0]),
        )
        .unwrap();
        subproof(
            &relations.vocabulary,
            &z,
            Fp2::ZERO,
            &[h.indicator()],
            |p, x| tv::eq(&tv_z, p).unwrap() * (TvFp2::ONE - mle_fp(&valid, p)) * x[0],
        )
        .unwrap();
        let target = Fp2::from(Fp::from_u64_reduced(count));
        assert!(
            subproof(&relations.count, &[], target, &[h.indicator()], |_, x| x[0]).is_err(),
            "real {real}"
        );
    }
    assert_eq!(reference(&[16, 5, 9, 3], 3_000_000_000), Some(2));
}

/// RFC-0050's carry widths, derived here from RFC-0058's coefficient bounds:
/// `K[j] = ⌊(A[j] + K[j−1])/2^16⌋`, `b[j] = max(1, bit_length(K[j]))`. The whole request
/// sequence, comparisons included, must be the one the verifier derives the same way (#160).
fn expected_requests(width: u8, sigma: u8) -> Vec<u32> {
    let k = usize::from(width).div_ceil(16);
    let m = k + 2;
    let full = (1u64 << 16) - 1;
    let top_bits = u32::from(width) - 16 * (k as u32 - 1);
    let top = (1u64 << top_bits) - 1;
    let sum_box = |h: usize| if h + 1 == k { top } else { full };
    let shifted = |j: usize| {
        if (2..k + 2).contains(&j) {
            sum_box(j - 2)
        } else {
            0
        }
    };
    let product = |j: usize| -> u64 {
        let mut total = 0;
        for i in 0..2 {
            for h in 0..k {
                if i + h == j {
                    total += full * sum_box(h);
                }
            }
        }
        total
    };
    let carries = |bounds: &[u64]| -> Vec<u32> {
        let mut carry = 0u64;
        bounds
            .iter()
            .map(|&a| {
                carry = (a + carry) >> 16;
                (64 - carry.leading_zeros()).max(1)
            })
            .collect()
    };
    let mut out = vec![16, 16];
    for _ in 0..2 {
        out.extend(std::iter::repeat_n(16, k - 1));
        out.push(top_bits);
    }
    let sides = [
        (
            (0..m).map(shifted).collect::<Vec<_>>(),
            (0..m).map(product).collect::<Vec<_>>(),
        ),
        (
            (0..m).map(|j| product(j) + u64::from(j == 0)).collect(),
            (0..m)
                .map(|j| shifted(j) + if j == 2 { 1 << sigma } else { 0 })
                .collect(),
        ),
    ];
    for (left, right) in sides {
        for bounds in [&left, &right] {
            out.extend(std::iter::repeat_n(16, m));
            out.extend(carries(bounds));
        }
        out.extend(std::iter::repeat_n(16, m));
        out.extend(std::iter::repeat_n(1, m));
    }
    out
}

#[test]
fn every_range_request_matches_the_widths_rfc_0050_derives() {
    for (vocab, sigma) in [
        (1, 0),
        (1, 15),
        (2, 16),
        (5, 16),
        (1, 31),
        (1, 32),
        (3, 32),
        (128_256, 16),
        (128_256, 32),
        (1 << 18, 32),
    ] {
        let g = gadget(vocab, sigma);
        let requests: Vec<u32> = g
            .constraints::<Fp>()
            .unwrap()
            .lookups
            .iter()
            .map(|l| l.table.0)
            .collect();
        assert_eq!(
            requests,
            expected_requests(g.width(), sigma),
            "vocab {vocab}, σ {sigma}"
        );
    }
    for width in 1..=51u8 {
        let (vocab, sigma) = if width <= 33 {
            (1, width - 1)
        } else {
            (1 << (width - 33), 32)
        };
        let g = gadget(vocab, sigma);
        let requests: Vec<u32> = g
            .constraints::<Fp>()
            .unwrap()
            .lookups
            .iter()
            .map(|l| l.table.0)
            .collect();
        assert_eq!(requests, expected_requests(width, sigma), "W = {width}");
    }
}

/// A row whose coefficients reach their bounds: `u_0 = Z_0 = 2^16 − 1` puts `P_0` at its bound
/// and the upper comparison's `X_0 = P_0 + ψ` one above it.
#[test]
fn a_row_at_its_coefficient_bounds_has_its_witness() {
    let g = gadget(2, 16);
    let weights = [65_536, 65_535];
    let e = padded(&weights);
    for u in [65_535u64, u64::from(u32::MAX)] {
        let token = reference(&weights, u).unwrap();
        let h = g.sample(&e, Fp::from_u64_reduced(u), true).unwrap();
        assert_eq!(h.indicator()[token], Fp::ONE);
        g.witness(&e, Fp::from_u64_reduced(u), &h, true).unwrap();
        let row = assignment(g.width(), u, sums(&weights, Some(token)), true);
        assert_eq!(failures(&g, &row), (vec![], vec![]), "u {u}");
    }
}

/// The suffix's dot product is load-bearing: the row accepts every prefix `B` from 14 to 18 here,
/// and a non-suffix `L′ = (0, 1, 1, 0)` gives `B = 14` through the prefix contraction. Only the
/// reduction of `L̃` to `ĥ` refuses it.
#[test]
fn only_the_suffix_reduction_refuses_a_non_suffix() {
    let g = gadget(4, 4);
    let weights = [16, 5, 9, 3];
    let e = padded(&weights);
    let u = 2_407_784_696u64;
    assert_eq!(reference(&weights, u), Some(1));
    for b in 14..=18 {
        let row = assignment(g.width(), u, (33, b, 5), true);
        assert_eq!(failures(&g, &row), (vec![], vec![]), "B = {b}");
    }
    let h = g.sample(&e, Fp::from_u64_reduced(u), true).unwrap();
    let forged = tessera_gadgets::CdfEvaluation {
        suffix: [0, 1, 1, 0]
            .iter()
            .map(|&x| Fp::from_u64_reduced(x))
            .collect(),
        sums: [33, 14, 5]
            .iter()
            .map(|&x| Fp::from_u64_reduced(x))
            .collect(),
    };
    let relations = g.relations::<Fp2>(&e, &h, &forged).unwrap();
    subproof(
        &relations.below,
        &[],
        Fp2::from(Fp::from_u64_reduced(14)),
        &[&e, &forged.suffix],
        |_, x| x[0] * x[1],
    )
    .unwrap();
    let rho: Vec<Fp2> = [7u64, 2]
        .map(|x| Fp2::from(Fp::from_u64_reduced(x)))
        .to_vec();
    let tv_rho: Vec<TvFp2> = rho.iter().map(|&x| tv_fp2(x)).collect();
    let reduced = g
        .suffix_weights(&rho)
        .unwrap()
        .iter()
        .zip(h.indicator())
        .fold(Fp2::ZERO, |acc, (&w, &x)| acc + w * Fp2::from(x));
    assert_ne!(tv_fp2(reduced), mle_fp(&forged.suffix, &tv_rho));
    let honest = g.evaluate(&e, &h).unwrap();
    assert_eq!(tv_fp2(reduced), mle_fp(&honest.suffix, &tv_rho));
}

/// The relations at a padded row prove zero sums, and at the largest vocabulary they prove the
/// partition `2^50`.
#[test]
fn the_relations_hold_at_a_padded_row_and_at_the_largest_vocabulary() {
    let g = gadget(6, 8);
    let e = padded(&[0; 6]);
    let h = g.sample(&e, Fp::from_u64_reduced(123), false).unwrap();
    let evaluated = g.evaluate(&e, &h).unwrap();
    assert!(evaluated.sums.iter().all(|&x| x == Fp::ZERO));
    let relations = g.relations::<Fp2>(&e, &h, &evaluated).unwrap();
    let valid: Vec<Fp> = (0..8)
        .map(|v| Fp::from_u64_reduced(u64::from(v < 6)))
        .collect();
    subproof(&relations.total, &[], Fp2::ZERO, &[&e], |r, x| {
        mle_fp(&valid, r) * x[0]
    })
    .unwrap();
    subproof(
        &relations.below,
        &[],
        Fp2::ZERO,
        &[&e, &evaluated.suffix],
        |_, x| x[0] * x[1],
    )
    .unwrap();
    subproof(
        &relations.selected,
        &[],
        Fp2::ZERO,
        &[h.indicator(), &e],
        |_, x| x[0] * x[1],
    )
    .unwrap();
    let sampled = g.sampled_relations::<Fp2>(&h).unwrap();
    subproof(&sampled.count, &[], Fp2::ZERO, &[h.indicator()], |_, x| {
        x[0]
    })
    .unwrap();

    let vocab = 1 << 18;
    let g = gadget(vocab, 32);
    let weights = vec![1u64 << 32; vocab];
    let e = padded(&weights);
    let u = 3_000_000_000u64;
    let token = reference(&weights, u).unwrap();
    let h = g.sample(&e, Fp::from_u64_reduced(u), true).unwrap();
    let evaluated = g.evaluate(&e, &h).unwrap();
    let relations = g.relations::<Fp2>(&e, &h, &evaluated).unwrap();
    let lift = |x: u128| Fp2::from(Fp::from_u64_reduced(x as u64));
    let (z, b, s) = sums(&weights, Some(token));
    assert_eq!(z, 1 << 50);
    subproof(&relations.total, &[], lift(z), &[&e], |_, x| x[0]).unwrap();
    subproof(
        &relations.below,
        &[],
        lift(b),
        &[&e, &evaluated.suffix],
        |_, x| x[0] * x[1],
    )
    .unwrap();
    subproof(
        &relations.selected,
        &[],
        lift(s),
        &[h.indicator(), &e],
        |_, x| x[0] * x[1],
    )
    .unwrap();
}
