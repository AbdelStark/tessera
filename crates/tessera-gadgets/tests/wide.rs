// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `RequantWide`'s uniqueness, RFC-0024 Lemma 11, over `F65537` (RFC-0009 'Testing Strategy',
//! RFC-0024 'Testing Strategy').
//!
//! Five configurations, as RFC-0024 'Why lookup 3' searched: one, two and four classes,
//! symmetric and asymmetric bounds, and `s = 0`, each the gadget the compiler selects for keys
//! of at most 14 bits. Every key and every class is tried.
//!
//! - **Exhaustion.** Of every `(z_lo', h0')` in `[0, 2^s) × [0, 2^t)`, with `h1_off` the virtual
//!   value they give and the `g'` and `y'` that best serve them, only the true digits pass. With
//!   them, only `g = clamp(h1, −2, 1)` and `y = rq(z, r_c, lo, hi)` do, each other value rejected
//!   at its own lookup.
//! - **Each range check is necessary.** `z_lo ± 2^s` and `h0 ± 2^t` pass every lookup but their
//!   own `Range` table's.
//! - **The odd-key witness.** `h0 − 2^{t−1}`, whose `h1_off` is the half-integer `h1_off + 1/2`,
//!   reaches the saturation table's odd key `2·h1_off + 1`. It is rejected at the top part's
//!   `Range { k }` alone. With that lookup removed it is accepted, and in every configuration it
//!   returns another `y` for some key: lookup 3 is what rejects it.

mod common;

use common::{first_rejected, rejected, rq, select_requant};
use tessera_compiler::Gadget;
use tessera_field::{F65537, PrimeField};
use tessera_gadgets::{Committed, Constraints, Requant, Term};
use tessera_tables::{GeneratedTable, Table};

type F = F65537;

/// A configuration: the key's width, the shifts, the bounds and the compiler's `B_max`.
struct Config {
    b: u32,
    shifts: &'static [u8],
    bounds: (i64, i64),
    b_max: u8,
}

/// RFC-0024's five configurations at keys of at most 14 bits, as the compiler selects them:
/// the expected `(s, t)` follow `s = min(r_min − 1, B_max)` and `t = (r_max − s) + ⌈log2 M⌉`.
const CONFIGS: [(Config, (u32, u32)); 5] = [
    // One class, symmetric bounds.
    (
        Config {
            b: 14,
            shifts: &[3],
            bounds: (-3, 3),
            b_max: 10,
        },
        (2, 3),
    ),
    // Two classes, asymmetric bounds.
    (
        Config {
            b: 14,
            shifts: &[3, 4],
            bounds: (-5, 2),
            b_max: 10,
        },
        (2, 5),
    ),
    // Four classes: the middle and top parts read one Range { 6 }.
    (
        Config {
            b: 13,
            shifts: &[2, 3, 4, 5],
            bounds: (-4, 4),
            b_max: 10,
        },
        (1, 6),
    ),
    // s = 0: no low part.
    (
        Config {
            b: 12,
            shifts: &[1, 2],
            bounds: (-2, 3),
            b_max: 9,
        },
        (0, 4),
    ),
    // One class, bounds that hold no negative value.
    (
        Config {
            b: 14,
            shifts: &[4],
            bounds: (0, 7),
            b_max: 10,
        },
        (3, 4),
    ),
];

/// The gadget, its constraints over `F65537`, and its tables.
struct Wide {
    gadget: Requant,
    constraints: Constraints<F>,
    s: u32,
    t: u32,
    k: u32,
}

impl Wide {
    fn new(config: &Config, expected: (u32, u32)) -> Self {
        let (lo, hi) = config.bounds;
        let (gadget, specs) = select_requant(config.b, config.shifts, lo, hi, config.b_max);
        let Gadget::RequantWide { split, middle, .. } = gadget else {
            panic!("RequantWide expected for {:?}", config.shifts);
        };
        let (s, t) = (u32::from(split), u32::from(middle));
        assert_eq!((s, t), expected, "{:?}", config.shifts);
        let gadget = Requant::new(&gadget, &specs).expect("the selection's gadget");
        let constraints = gadget.constraints::<F>();
        Self {
            gadget,
            constraints,
            s,
            t,
            k: config.b - s - t,
        }
    }

    fn tables(&self) -> Vec<&GeneratedTable> {
        self.gadget.tables()
    }

    /// The index of the lookup into `table`'s role: 0 `Range { s }`, 1 `Range { t }`, 2
    /// `Range { k }`, 3 the saturation, 4 the output, shifted down when there is no low part.
    fn lookup(&self, role: usize) -> usize {
        match self.s {
            0 => role - 1,
            _ => role,
        }
    }
}

/// An assignment of `RequantWide`'s terms.
#[derive(Clone, Copy)]
struct Assignment {
    z: F,
    c: F,
    z_lo: F,
    h0: F,
    g: F,
    y: F,
}

impl Assignment {
    fn value(self) -> impl FnMut(Term) -> F {
        move |term| match term {
            Term::Input => self.z,
            Term::Class => self.c,
            Term::Committed(Committed::ZLo) => self.z_lo,
            Term::Committed(Committed::H0) => self.h0,
            Term::Committed(Committed::G) => self.g,
            Term::Committed(Committed::Y) => self.y,
            Term::Committed(other) => panic!("RequantWide commits no {other:?}"),
        }
    }
}

/// The value of the row `key` names in `table`, if it names one.
fn row_value(table: &GeneratedTable, key: F) -> Option<F> {
    let row = key.to_canonical_u64();
    (row < table.row_count()).then(|| {
        let (_, value) = table.row(row).expect("a row");
        F::from_i64(i64::try_from(value.to_centered_i128()).expect("a true integer"))
    })
}

/// The table lookup `index` reads.
fn table_of<'a>(wide: &Wide, tables: &[&'a GeneratedTable], index: usize) -> &'a GeneratedTable {
    let id = wide.constraints.lookups[index].table;
    tables.iter().find(|t| t.spec().id == id).expect("a table")
}

/// `assignment` with the `y'` that best serves it: the output table's value at lookup 5's key,
/// or `None` when that key is no row, so that no `y'` passes.
fn serve_y(wide: &Wide, tables: &[&GeneratedTable], assignment: Assignment) -> Option<Assignment> {
    let high = wide.lookup(4);
    let key = wide.constraints.lookups[high]
        .key
        .evaluate(assignment.value());
    let y = row_value(table_of(wide, tables, high), key)?;
    Some(Assignment { y, ..assignment })
}

/// `(z, c, z_lo', h0')` with the `g'` and `y'` that best serve them: the saturation table's
/// value at lookup 4's key, then the output table's at lookup 5's, when those keys are rows.
/// `None` when either key is no row, so that no `g'` or `y'` passes.
fn best(wide: &Wide, tables: &[&GeneratedTable], partial: Assignment) -> Option<Assignment> {
    let sign = wide.lookup(3);
    let key = wide.constraints.lookups[sign].key.evaluate(partial.value());
    let g = row_value(table_of(wide, tables, sign), key)?;
    serve_y(wide, tables, Assignment { g, ..partial })
}

/// Every key and class of a configuration, with its true digits and output.
fn each_key(config: &Config, wide: &Wide, mut visit: impl FnMut(i64, usize, Assignment, i64)) {
    let half = 1i64 << (config.b - 1);
    let (lo, hi) = config.bounds;
    for z in -half..half {
        let z_off = z + half;
        let (z_lo, w_off) = (z_off % (1 << wide.s), z_off >> wide.s);
        let h0 = w_off % (1 << wide.t);
        let h1 = (z >> (wide.s + wide.t)).clamp(-2, 1);
        for (class, &r) in config.shifts.iter().enumerate() {
            let y = rq(z, r, lo, hi);
            let honest = Assignment {
                z: F::from_i64(z),
                c: F::from_u64_reduced(class as u64),
                z_lo: F::from_i64(z_lo),
                h0: F::from_i64(h0),
                g: F::from_i64(h1),
                y: F::from_i64(y),
            };
            visit(z, class, honest, h0);
        }
    }
}

#[test]
fn only_the_true_digits_pass_by_exhaustion() {
    for (config, expected) in &CONFIGS {
        let wide = Wide::new(config, *expected);
        let tables = wide.tables();
        let (lo, hi) = config.bounds;
        each_key(config, &wide, |z, class, honest, _| {
            // The honest witness passes every lookup.
            assert_eq!(
                rejected(&wide.constraints, &tables, honest.value()),
                Vec::<usize>::new(),
                "z = {z}, class {class}"
            );
            // Of every pair of digits, only the true one passes, with any `g'` and `y'`.
            for z_lo in 0..1i64 << wide.s {
                for h0 in 0..1i64 << wide.t {
                    let partial = Assignment {
                        z_lo: F::from_i64(z_lo),
                        h0: F::from_i64(h0),
                        ..honest
                    };
                    let passes = best(&wide, &tables, partial).is_some_and(|candidate| {
                        first_rejected(&wide.constraints, &tables, candidate.value()).is_none()
                    });
                    let truth = F::from_i64(z_lo) == honest.z_lo && F::from_i64(h0) == honest.h0;
                    assert_eq!(passes, truth, "z = {z}, class {class}, ({z_lo}, {h0})");
                }
            }
            // With the true digits, only the true `g` and `y` pass, each other value rejected at
            // its own lookup.
            for g in -2..=1i64 {
                let g = F::from_i64(g);
                let candidate = serve_y(&wide, &tables, Assignment { g, ..honest })
                    .expect("a saturation value keeps lookup 5's key in its table");
                let found = rejected(&wide.constraints, &tables, candidate.value());
                let expected: Vec<usize> = match g == honest.g {
                    true => vec![],
                    false => vec![wide.lookup(3)],
                };
                assert_eq!(found, expected, "z = {z}, class {class}, g' = {g:?}");
            }
            for y in lo - 2..=hi + 2 {
                let candidate = Assignment {
                    y: F::from_i64(y),
                    ..honest
                };
                let found = rejected(&wide.constraints, &tables, candidate.value());
                let expected: Vec<usize> = match F::from_i64(y) == honest.y {
                    true => vec![],
                    false => vec![wide.lookup(4)],
                };
                assert_eq!(found, expected, "z = {z}, class {class}, y' = {y}");
            }
        });
    }
}

#[test]
fn each_range_check_rejects_the_digits_shifted_past_it() {
    for (config, expected) in &CONFIGS {
        let wide = Wide::new(config, *expected);
        let tables = wide.tables();
        let mut shifted = [0usize; 2];
        each_key(config, &wide, |z, class, honest, _| {
            // `h0 ± 2^t`, which the virtual top part carries, and `z_lo ± 2^s` with `h0 ∓ 1`,
            // which the virtual `w_off` carries: every virtual value stays an integer.
            let mut alternatives = vec![
                (
                    wide.lookup(1),
                    Assignment {
                        h0: honest.h0 + F::from_i64(1 << wide.t),
                        ..honest
                    },
                ),
                (
                    wide.lookup(1),
                    Assignment {
                        h0: honest.h0 - F::from_i64(1 << wide.t),
                        ..honest
                    },
                ),
            ];
            if wide.s >= 1 {
                for d in [1i64, -1] {
                    alternatives.push((
                        0,
                        Assignment {
                            z_lo: honest.z_lo + F::from_i64(d << wide.s),
                            h0: honest.h0 - F::from_i64(d),
                            ..honest
                        },
                    ));
                }
            }
            for (range, partial) in alternatives {
                let Some(candidate) = best(&wide, &tables, partial) else {
                    // Its top part left its table; the range check is not alone.
                    continue;
                };
                let found = rejected(&wide.constraints, &tables, candidate.value());
                if found.len() == 1 {
                    assert_eq!(found, [range], "z = {z}, class {class}");
                    shifted[usize::from(range != 0 || wide.s == 0)] += 1;
                } else {
                    assert!(found.contains(&range), "z = {z}, class {class}: {found:?}");
                }
            }
        });
        // Each check is the only one left against the shifted digits, for some key.
        assert!(shifted[1] > 0, "{:?}: Range {{ t }}", config.shifts);
        assert!(
            wide.s == 0 || shifted[0] > 0,
            "{:?}: Range {{ s }}",
            config.shifts
        );
    }
}

#[test]
fn the_odd_key_witness_is_rejected_by_the_top_range_check_alone() {
    for (config, expected) in &CONFIGS {
        let wide = Wide::new(config, *expected);
        let tables = wide.tables();
        let top = wide.lookup(2);
        let mut without_top = wide.constraints.clone();
        without_top.lookups.remove(top);
        let (mut tried, mut other_output) = (0, 0);
        each_key(config, &wide, |z, class, honest, h0| {
            if h0 < 1 << (wide.t - 1) {
                return;
            }
            // `h0 − 2^{t−1}` gives `h1_off + 1/2`, whose double is the odd key `2·h1_off + 1`.
            let partial = Assignment {
                h0: honest.h0 - F::from_i64(1 << (wide.t - 1)),
                ..honest
            };
            let candidate =
                best(&wide, &tables, partial).expect("the odd key is a row of the saturation");
            assert_eq!(
                rejected(&wide.constraints, &tables, candidate.value()),
                [top],
                "z = {z}, class {class}"
            );
            // Without lookup 3, it passes: lookups 1, 2, 4 and 5 hold.
            assert_eq!(
                rejected(&without_top, &tables, candidate.value()),
                Vec::<usize>::new(),
                "z = {z}, class {class}"
            );
            tried += 1;
            other_output += usize::from(candidate.y != honest.y);
        });
        assert!(tried > 0, "{:?}", config.shifts);
        assert!(
            other_output > 0,
            "{:?}: some key's odd-key witness returns another y",
            config.shifts
        );
        assert!(wide.k >= 3, "RequantWide is selected only with k ≥ 3");
    }
}
