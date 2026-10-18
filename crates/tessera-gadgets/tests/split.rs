// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `RequantSplit`'s uniqueness, RFC-0009 Lemma 2, over `F65537` and Goldilocks.
//!
//! - **Exhaustion** (RFC-0009 'Testing Strategy'): for keys of up to 12 bits, every split point
//!   `s` Lemma 2 allows below the rounding bit, every key and class, and every `z_lo'` that
//!   passes `Range { s }`, only the true `z_lo` passes the high lookup, whatever `y'` it
//!   commits; with it, only the true `y`, which is `rq(z, r_c, lo, hi)`.
//! - **Alternative witnesses** `z_lo ± 1` and `z_lo ± 2^s`, each with the `y'` that best serves
//!   it, are rejected at exactly the lookups their integer arithmetic predicts: `z_lo ± 1` at
//!   the high lookup, whose key is no integer, and `z_lo ± 2^s` at `Range { s }` alone when its
//!   high key is a row. Without `Range { s }`, those are accepted: the check is necessary.
//! - **`n·2^b < P`** (Lemma 2's precondition). Four classes at 62 bits break it over Goldilocks:
//!   `z_lo ± 1` at either end of the key's range wraps the high key to the table's other end, and
//!   at shifts 33 to 36 passes with the other saturated output. The constructor refuses that
//!   split, the compiler takes `RequantWide`, which rejects the same low part, and three classes
//!   at 62 bits reject it. Over `F65537`, by exhaustion, the bound is exact: past it second
//!   witnesses pass, below it none do.

mod common;

use common::{Fp, SplitMix64, first_rejected, in_table, keys, rejected, rq, select_requant};
use tessera_compiler::Gadget;
use tessera_field::{F65537, Field, PrimeField};
use tessera_gadgets::{Committed, Constraints, GadgetError, Requant, Term};
use tessera_poly::PolyField;
use tessera_tables::{GeneratedTable, Table, TableId, TableKind, TableSpec};

/// `RequantSplit` at split point `s` for keys of `b` bits, built by hand: `Range { s }` and
/// `Requant { b − s, r_c − s, lo, hi }`, which Lemma 2 allows for any `1 ≤ s ≤ r_min − 1`.
fn split(b: u8, s: u8, shifts: &[u8], (lo, hi): (i32, i32)) -> Requant {
    let low = GeneratedTable::new(TableId(0), TableKind::Range { bits: s }).expect("Range { s }");
    let high = GeneratedTable::new(
        TableId(1),
        TableKind::Requant {
            b: b - s,
            r_by_class: shifts.iter().map(|r| r - s).collect(),
            lo,
            hi,
        },
    )
    .expect("the reduced table");
    let specs: Vec<TableSpec> = [low, high].iter().map(|t| t.spec().clone()).collect();
    let gadget = Gadget::RequantSplit {
        split: s,
        low: TableId(0),
        high: TableId(1),
    };
    Requant::new(&gadget, &specs).expect("a split gadget")
}

/// The value of the row `key` names in `table`, if it names one.
fn row_value<F: PrimeField>(table: &GeneratedTable, key: F) -> Option<F> {
    let row = key.to_canonical_u64();
    (row < table.row_count()).then(|| {
        let (_, value) = table.row(row).expect("a row");
        F::from_i64(i64::try_from(value.to_centered_i128()).expect("a true integer"))
    })
}

/// The assignment `(z, c, z_lo, y)`.
fn assign<F: PrimeField>(z: F, c: F, z_lo: F, y: F) -> impl FnMut(Term) -> F {
    move |term| match term {
        Term::Input => z,
        Term::Class => c,
        Term::Committed(Committed::ZLo) => z_lo,
        Term::Committed(Committed::Y) => y,
        Term::Committed(other) => panic!("RequantSplit commits no {other:?}"),
    }
}

/// The high lookup's key at `(z, c, z_lo)`; it reads neither `y` nor anything else.
fn high_key<F: PrimeField + PolyField<Base = F>>(
    constraints: &Constraints<F>,
    z: F,
    c: F,
    z_lo: F,
) -> F {
    constraints.lookups[1]
        .key
        .evaluate(assign(z, c, z_lo, F::ZERO))
}

/// Lemma 2 by exhaustion over `F`: every key and class, and every `z_lo'` in `[0, 2^s)`.
fn exhaust<F: PrimeField + PolyField<Base = F>>(
    gadget: &Requant,
    (b, s): (u32, u32),
    shifts: &[u8],
    (lo, hi): (i64, i64),
) {
    let constraints = gadget.constraints::<F>();
    let tables = gadget.tables();
    let high = tables[1];
    let half = 1i64 << (b - 1);
    for z in -half..half {
        for (class, &r) in shifts.iter().enumerate() {
            let (zf, c) = (F::from_i64(z), F::from_u64_reduced(class as u64));
            let honest = (z + half) % (1 << s);
            let y = rq(z, r, lo, hi);
            for candidate in 0..1i64 << s {
                // The best `y'` for this `z_lo'`: the high table's value at its key, if a row.
                let key = high_key(&constraints, zf, c, F::from_i64(candidate));
                let passes = row_value(high, key).is_some_and(|y| {
                    first_rejected(
                        &constraints,
                        &tables,
                        assign(zf, c, F::from_i64(candidate), y),
                    )
                    .is_none()
                });
                assert_eq!(
                    passes,
                    candidate == honest,
                    "b = {b}, s = {s}, z = {z}, class {class}, z_lo' = {candidate}"
                );
            }
            // The true `z_lo` passes with `y = rq(z, r_c, lo, hi)` alone.
            for candidate in lo - 2..=hi + 2 {
                let found = rejected(
                    &constraints,
                    &tables,
                    assign(zf, c, F::from_i64(honest), F::from_i64(candidate)),
                );
                let expected: &[usize] = if candidate == y { &[] } else { &[1] };
                assert_eq!(found, expected, "z = {z}, class {class}, y' = {candidate}");
            }
        }
    }
}

/// Every split point Lemma 2 allows for these shifts, `1 ≤ s ≤ min(r_min − 1, b − 1)`, by
/// exhaustion over both fields.
fn every_split(b: u8, shifts: &[u8], bounds: (i32, i32)) {
    let least = shifts.iter().copied().min().expect("a shift");
    for s in 1..least.min(b) {
        let gadget = split(b, s, shifts, bounds);
        let wide = (i64::from(bounds.0), i64::from(bounds.1));
        let parts = (u32::from(b), u32::from(s));
        exhaust::<F65537>(&gadget, parts, shifts, wide);
        exhaust::<Fp>(&gadget, parts, shifts, wide);
    }
}

#[test]
fn only_the_true_low_part_passes_by_exhaustion() {
    // One class, every split from 1 to 8; two classes and four, with asymmetric bounds; a small
    // shift.
    every_split(12, &[9], (-128, 127));
    every_split(12, &[7, 9], (-50, 77));
    every_split(10, &[4, 5, 6, 7], (-8, 8));
    every_split(11, &[3], (-3, 3));
    // The split the compiler selects at B_max = 8 for 12 bits and a shift of 9 (RFC-0028).
    let (gadget, specs) = select_requant(12, &[9], -128, 127, 8);
    assert!(matches!(gadget, Gadget::RequantSplit { split: 6, .. }));
    let selected = Requant::new(&gadget, &specs).expect("the selection's gadget");
    exhaust::<F65537>(&selected, (12, 6), &[9], (-128, 127));
}

/// The lookups an alternative `z_lo' = z_lo + d` fails, by integer arithmetic: `Range { s }`
/// unless `0 ≤ z_lo' < 2^s`, and the high lookup unless `d` is a multiple of `2^s` and the key
/// `z_hi − d/2^s + 2^{b−s}·c` lies in the table's `n·2^{b−s}` rows.
fn predicted(z_off: i64, s: u32, d: i64, b: u32, class: i64, classes: i64) -> Vec<usize> {
    let z_lo = z_off % (1 << s);
    let mut fails = Vec::new();
    if !(0..1 << s).contains(&(z_lo + d)) {
        fails.push(0);
    }
    let reduced = 1i64 << (b - s);
    let row = (z_off >> s) - d / (1 << s) + reduced * class;
    if d % (1 << s) != 0 || !(0..classes * reduced).contains(&row) {
        fails.push(1);
    }
    fails
}

/// `z_lo ± 1` and `z_lo ± 2^s` at every key given, each with its best `y'`, are rejected at
/// exactly the predicted lookups; without `Range { s }`, `z_lo ± 2^s` passes where its high key
/// is a row.
fn alternatives(gadget: &Requant, (b, s): (u32, u32), keys: &[i64], classes: usize) -> usize {
    let constraints = gadget.constraints::<Fp>();
    let tables = gadget.tables();
    let high = tables[1];
    let mut without_range = constraints.clone();
    without_range.lookups.remove(0);
    let half = 1i64 << (b - 1);
    let mut accepted_without = 0;
    for &z in keys {
        for class in 0..classes {
            let (zf, c) = (Fp::from_i64(z), Fp::from_u64_reduced(class as u64));
            let z_off = z + half;
            for d in [1, -1, 1 << s, -(1 << s)] {
                let z_lo = Fp::from_i64(z_off % (1 << s) + d);
                let key = high_key(&constraints, zf, c, z_lo);
                // The value the high table holds at that key; when it holds none, no `y'` passes.
                let y = row_value(high, key).unwrap_or(Fp::ZERO);
                let found = rejected(&constraints, &tables, assign(zf, c, z_lo, y));
                let expected = predicted(z_off, s, d, b, class as i64, classes as i64);
                assert!(!found.is_empty(), "z = {z}, d = {d}: accepted");
                assert_eq!(found, expected, "z = {z}, class {class}, d = {d}");
                // Without the low part's range check, only the high lookup is left.
                let left = rejected(&without_range, &tables, assign(zf, c, z_lo, y));
                if !expected.contains(&1) {
                    assert!(left.is_empty(), "z = {z}, d = {d}");
                    assert!(in_table(high, key, Some(y)));
                    accepted_without += 1;
                }
            }
        }
    }
    accepted_without
}

#[test]
fn alternative_low_parts_are_rejected_at_the_predicted_lookup() {
    let mut rng = SplitMix64::new(0x118_0102);
    // Every key of 12 bits with two classes, at s = 6.
    let gadget = split(12, 6, &[7, 9], (-50, 77));
    let all: Vec<i64> = (-2048..2048).collect();
    let accepted = alternatives(&gadget, (12, 6), &all, 2);
    assert!(accepted > 0, "z_lo ± 2^s passes without Range {{ s }}");
    // Production: the balanced split of a 48-bit key (s = 24) and of a 62-bit one (s = 30).
    for (b, r) in [(48u32, 40u8), (62, 31)] {
        let (selected, specs) = select_requant(b, &[r], -128, 127, 32);
        let Gadget::RequantSplit { split: s, .. } = selected else {
            panic!("RequantSplit expected");
        };
        let gadget = Requant::new(&selected, &specs).expect("the selection's gadget");
        let keys = keys(&mut rng, b, 2_000);
        let accepted = alternatives(&gadget, (b, u32::from(s)), &keys, 1);
        assert!(accepted > 0, "b = {b}");
    }
}

/// The low parts one step from the true ones at both ends of a 62-bit key's range, as
/// `(class, z, step)`: `z_lo + 1` at `z = −2^61 + d` in class 0, where `z_lo = d`, and `z_lo − 1`
/// at `z = 2^61 − d` in class `top`, where `z_lo = 2^32 − d`.
fn ends(top: u8) -> Vec<(u8, i64, i64)> {
    let half = 1i64 << 61;
    let bottom = [0, 1, 12_345, (1 << 32) - 3, (1 << 32) - 2].map(|d| (0, -half + d, 1));
    let upper = [1, 2, 777, (1 << 32) - 2, (1 << 32) - 1].map(|d| (top, half - d, -1));
    bottom.into_iter().chain(upper).collect()
}

/// RFC-0009 Lemma 2 compares `z_lo + 2^s·j` with `z_off + 2^b·c`, both in `[0, n·2^b)`, so it
/// needs `n·2^b < P`. Four classes at 62 bits break it, `4·2^62 = 2^64 > P`, at the split point
/// the compiler would take, `s = 32`. Since `2^32·(2^32 − 1) = P − 1`, with shifts 33 to 36 and
/// INT8 bounds:
///
/// - in class 0 at `z = −2^61 + d`, `z_lo + 1` makes the high key `2^32 − 1`, the table's last
///   row, class 3's largest key, whose output is `hi` where the true one is `lo`;
/// - in class 3 at `z = 2^61 − d`, `z_lo − 1` makes it 0, class 0's least key, whose output is
///   `lo` where the true one is `hi`.
///
/// The constructor refuses that split, and the compiler takes `RequantWide`, which rejects the
/// same low parts. Three classes at 62 bits, `3·2^62 < P`, reject them too.
#[test]
fn four_classes_at_62_bits_wrap_past_p_without_the_precondition() {
    let shifts = [33u8, 34, 35, 36];
    let (lo, hi) = (-128, 127);
    let half = 1i64 << 61;
    let low_bits = (1i64 << 32) - 1;
    // The compiler's split of three classes at 62 bits, at the balanced s = ⌈(62 + 2)/2⌉ = 32.
    let (selected, specs) = select_requant(62, &shifts[..3], lo, hi, 32);
    assert!(
        matches!(selected, Gadget::RequantSplit { split: 32, .. }),
        "{selected:?}"
    );
    let three = Requant::new(&selected, &specs).expect("3·2^62 < P");
    let constraints = three.constraints::<Fp>();
    let tables = three.tables();
    // The split of four classes has the same forms, which read the classes only through their
    // class term, its high lookup reading the reduced table of four classes. The constructor
    // refuses it; below it, the forms and that table are what a prover would be held to.
    let four = GeneratedTable::new(
        tables[1].spec().id,
        TableKind::Requant {
            b: 30,
            r_by_class: shifts.iter().map(|r| r - 32).collect(),
            lo: -128,
            hi: 127,
        },
    )
    .expect("the reduced table of four classes");
    assert_eq!(
        Requant::new(&selected, &[tables[0].spec().clone(), four.spec().clone()]).err(),
        Some(GadgetError::Shape {
            rule: "a split key of n classes has n·2^b < P (RFC-0009 Lemma 2)"
        })
    );
    let broken = [tables[0], &four];
    for (class, z, step) in ends(3) {
        let (zf, c) = (Fp::from_i64(z), Fp::from_u64_reduced(u64::from(class)));
        let z_lo = (z + half) & low_bits;
        let y = rq(z, shifts[usize::from(class)], lo, hi);
        let honest = assign(zf, c, Fp::from_i64(z_lo), Fp::from_i64(y));
        assert!(
            rejected(&constraints, &broken, honest).is_empty(),
            "z = {z}"
        );
        // The second witness: the high key wraps to the table's other end, whose row holds the
        // other saturated output.
        let alternative = Fp::from_i64(z_lo + step);
        let key = high_key(&constraints, zf, c, alternative);
        let (row, outputs) = match class {
            0 => ((1u64 << 32) - 1, (lo, hi)),
            _ => (0, (hi, lo)),
        };
        assert_eq!(key.to_canonical_u64(), row, "z = {z}, class {class}");
        let y_alt = row_value(&four, key).expect("a row of the table");
        assert_eq!(
            (
                y,
                i64::try_from(y_alt.to_centered_i128()).expect("an output")
            ),
            outputs,
            "z = {z}, class {class}"
        );
        let second = assign(zf, c, alternative, y_alt);
        assert!(
            rejected(&constraints, &broken, second).is_empty(),
            "z = {z}, class {class}: z_lo {step:+} rejected"
        );
    }
    // Three classes: the same low parts, at class 0's bottom and class 2's top, name no row of
    // their table, whatever y' they commit.
    for (class, z, step) in ends(2) {
        let (zf, c) = (Fp::from_i64(z), Fp::from_u64_reduced(u64::from(class)));
        let alternative = Fp::from_i64(((z + half) & low_bits) + step);
        let key = high_key(&constraints, zf, c, alternative);
        assert_eq!(row_value(tables[1], key), None, "z = {z}, class {class}");
        let found = rejected(&constraints, &tables, assign(zf, c, alternative, Fp::ZERO));
        assert_eq!(found, [1], "z = {z}, class {class}");
    }
    // The compiler's RequantWide (RFC-0024): s = min(33 − 1, 32) = 32,
    // t = (36 − 32) + ⌈log2 128⌉ = 11 and k = 62 − 32 − 11 = 19. The same low part leaves
    // w_off = (z_off − z_lo')·2^{−32} no integer below 2^30, so with every middle digit h0',
    // h1_off misses Range { 19 }: lookup 2, which reads neither g nor y.
    let (selected, specs) = select_requant(62, &shifts, lo, hi, 32);
    assert!(
        matches!(
            selected,
            Gadget::RequantWide {
                split: 32,
                middle: 11,
                ..
            }
        ),
        "{selected:?}"
    );
    let wide = Requant::new(&selected, &specs).expect("Lemma 11's preconditions hold");
    let constraints = wide.constraints::<Fp>();
    let tables = wide.tables();
    for (class, z, step) in ends(3) {
        let (zf, c) = (Fp::from_i64(z), Fp::from_u64_reduced(u64::from(class)));
        let witness = wide
            .witness(&[z], Some(&[class]))
            .expect("an honest witness");
        assert_eq!(
            witness.tensor(Committed::Y),
            Some(&[rq(z, shifts[usize::from(class)], lo, hi)][..])
        );
        let entry = |tensor| Fp::from_i64(witness.tensor(tensor).expect("a tensor")[0]);
        let honest = |term| match term {
            Term::Input => zf,
            Term::Class => c,
            Term::Committed(tensor) => entry(tensor),
        };
        assert!(
            rejected(&constraints, &tables, honest).is_empty(),
            "z = {z}"
        );
        let alternative = Fp::from_i64(((z + half) & low_bits) + step);
        for h0 in 0..1u64 << 11 {
            let found = first_rejected(&constraints, &tables, |term| match term {
                Term::Input => zf,
                Term::Class => c,
                Term::Committed(Committed::ZLo) => alternative,
                Term::Committed(Committed::H0) => Fp::from_u64_reduced(h0),
                Term::Committed(_) => Fp::ZERO,
            });
            assert_eq!(found, Some(2), "z = {z}, class {class}, h0' = {h0}");
        }
    }
}

/// The second witnesses of `gadget` over `F`, by exhaustion: of every key, class and
/// `z_lo' ≠ z_lo` in `[0, 2^s)`, each with the `y'` its high key's row holds, those that pass
/// both lookups, and how many of them return another output than `rq(z, r_c, lo, hi)`.
fn second_witnesses<F: PrimeField + PolyField<Base = F>>(
    gadget: &Requant,
    (b, s): (u32, u32),
    shifts: &[u8],
    (lo, hi): (i64, i64),
) -> (usize, usize) {
    let constraints = gadget.constraints::<F>();
    let tables = gadget.tables();
    let half = 1i64 << (b - 1);
    let (mut second, mut changed) = (0, 0);
    for z in -half..half {
        for (class, &r) in shifts.iter().enumerate() {
            let (zf, c) = (F::from_i64(z), F::from_u64_reduced(class as u64));
            let honest = (z + half) % (1 << s);
            for candidate in (0..1i64 << s).filter(|&z_lo| z_lo != honest) {
                let z_lo = F::from_i64(candidate);
                let key = high_key(&constraints, zf, c, z_lo);
                let Some(y) = row_value(tables[1], key) else {
                    continue;
                };
                if first_rejected(&constraints, &tables, assign(zf, c, z_lo, y)).is_none() {
                    second += 1;
                    changed += usize::from(y.to_centered_i128() != i128::from(rq(z, r, lo, hi)));
                }
            }
        }
    }
    (second, changed)
}

#[test]
fn over_f65537_n_2_b_below_p_is_the_exact_bound() {
    // P = 2^16 + 1 (RFC-0009 'Testing Strategy'). Three classes at 15 bits, 3·2^15 > P: class 0's
    // bottom and class 2's top wrap to each other, with another output. The same gadget over
    // Goldilocks has no second witness. Four classes at 14 bits and two at 15, 2^16 < P, have
    // none either, at the least split point and at the most below the rounding bit.
    let shifts = [5u8, 6, 7, 8];
    let bounds = (-8, 8);
    let wide = (i64::from(bounds.0), i64::from(bounds.1));
    for (b, n, wraps) in [(15u8, 3usize, true), (14, 4, false), (15, 2, false)] {
        for s in [1u8, 4] {
            let gadget = split(b, s, &shifts[..n], bounds);
            let parts = (u32::from(b), u32::from(s));
            let (second, changed) = second_witnesses::<F65537>(&gadget, parts, &shifts[..n], wide);
            let context =
                format!("b = {b}, n = {n}, s = {s}: {second} pass, {changed} with another y");
            if wraps {
                assert!(changed > 0, "{context}");
                let (over_goldilocks, _) =
                    second_witnesses::<Fp>(&gadget, parts, &shifts[..n], wide);
                assert_eq!(over_goldilocks, 0, "{context}, over Goldilocks");
            } else {
                assert_eq!(second, 0, "{context}");
            }
        }
    }
}
